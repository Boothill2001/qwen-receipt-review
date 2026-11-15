"""Prepare an address-only WildReceipt evaluation without extracting an archive."""

import hashlib
import json
import unicodedata
from collections import Counter
from io import BytesIO
from pathlib import PurePosixPath

from PIL import Image, ImageOps

from address_protocol import ADDRESS_CASES
from benchmark_receipts import dataset

SOURCE = "https://download.openmmlab.com/mmocr/data/wildreceipt.tar"
RULE = "wildreceipt-address-v1"
SIZES = {"dev": 10, "holdout": 50}


def canonical_group(value):
    return "".join(c for c in unicodedata.normalize("NFKC", value).casefold() if c.isalnum())


def reading_order(annotations, label):
    boxes = []
    for annotation in annotations:
        if annotation["label"] != label or not annotation["text"].strip():
            continue
        points = annotation["box"]
        if len(points) != 8:
            raise ValueError("Expected a four-point text polygon")
        y1, y2 = min(points[1::2]), max(points[1::2])
        boxes.append(((y1 + y2) / 2, min(points[::2]), max(1, y2 - y1), annotation["text"].strip()))
    lines = []
    for center, x, height, text in sorted(boxes):
        if not lines or abs(center - lines[-1][0]) > 0.45 * min(height, lines[-1][1]):
            lines.append((center, height, []))
        lines[-1][2].append((x, text))
    return " ".join(text for _, _, line in lines for _, text in sorted(line))


def fingerprint(data):
    with Image.open(BytesIO(data)) as source:
        image = ImageOps.exif_transpose(source)
        ratio = image.width / image.height
        pixels = list(ImageOps.grayscale(image).resize((9, 8), Image.Resampling.LANCZOS).getdata())
    dhash = sum(int(pixels[y * 9 + x] > pixels[y * 9 + x + 1]) << (y * 8 + x)
                for y in range(8) for x in range(8))
    return hashlib.sha256(data).hexdigest(), dhash, ratio


def is_duplicate(left, right):
    return left[0] == right[0] or (
        abs(left[2] / right[2] - 1) <= 0.05 and (left[1] ^ right[1]).bit_count() <= 3
    )


def read_bytes(archive, member):
    with archive.extractfile(archive.getmember(member)) as stream:
        return stream.read()


def wild_candidates(archive, split):
    classes = next(member for member in archive.getmembers()
                   if member.isfile() and PurePosixPath(member.name).name == "class_list.txt")
    root = PurePosixPath(classes.name).parent
    labels = {}
    for line in read_bytes(archive, classes.name).decode("utf-8-sig").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit():
            labels[parts[1].casefold()] = int(parts[0])
    address_id, merchant_id = labels["store_addr_value"], labels["store_name_value"]
    name = str(root / f"{split}.txt")
    prefix = f"wildreceipt:{split}:"
    rows = []
    counts = Counter()
    for line in read_bytes(archive, name).decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        address = reading_order(row["annotations"], address_id)
        merchant = canonical_group(reading_order(row["annotations"], merchant_id))
        if not address or not merchant:
            counts["missing_address_or_merchant"] += 1
            continue
        relative = PurePosixPath(row["file_name"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid image member path")
        member = str(root / relative)
        if not archive.getmember(member).isfile():
            raise ValueError("Expected a regular image member")
        key = prefix + relative.as_posix()
        rows.append({"id": key, "member": member, "merchant_group": merchant,
                     "annotation_address": address})
    return sorted(rows, key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest()), counts


def select_wild(archive, split, count, seen_groups, seen_images):
    candidates, rejects = wild_candidates(archive, split)
    output = []
    for row in candidates:
        if row["merchant_group"] in seen_groups:
            rejects["repeated_merchant"] += 1
            continue
        candidate = fingerprint(read_bytes(archive, row["member"]))
        if any(is_duplicate(candidate, prior) for prior in seen_images):
            rejects["duplicate_or_near_duplicate"] += 1
            continue
        output.append({**row, "image_sha256": candidate[0]})
        seen_images.append(candidate)
        seen_groups.add(row["merchant_group"])
        if len(output) == count:
            break
    if len(output) != count:
        raise ValueError(f"Only {len(output)}/{count} eligible {split} receipts: {dict(rejects)}")
    return output, dict(rejects)


def cohorts(wild, sroie):
    originals, old_images = {}, []
    for stage, cohort in (("dev", "address-dev10"), ("holdout", "pilot")):
        receipts, labels, *_ = dataset(sroie, cohort)
        originals[stage] = []
        for receipt in receipts:
            if stage == "dev" and ADDRESS_CASES[receipt.key] != "control":
                continue
            digest = fingerprint(sroie.read(receipt.image))
            old_images.append(digest)
            originals[stage].append({"id": f"sroie:{receipt.key}", "member": receipt.image,
                                     "address": labels[receipt.key]["address"], "image_sha256": digest[0]})
    merchants = set()
    result = {}
    for stage, split in (("dev", "train"), ("holdout", "test")):
        rows, rejects = select_wild(wild, split, SIZES[stage], merchants, old_images)
        result[stage] = {"wildreceipt": rows, "sroie": originals[stage], "excluded_before_quota": rejects}
    return result
