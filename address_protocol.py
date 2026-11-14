"""Frozen inputs for the paired address prompt experiment."""

from receipt_pipeline import PROMPT, PROMPT_VERSION

ADDRESS_PROMPT_VERSION = "receipt-address-v2"
ADDRESS_CASES = {
    "X51005453802": "format", "X51008145504": "format", "X51006414394": "format",
    "X51006008086": "content", "X51008123586": "content",
    "X51006556818": "content", "X51005568913": "content",
    "X51006388065": "control", "X51006441474": "control", "X51008164997": "control",
}
PROMPTS = {
    PROMPT_VERSION: PROMPT,
    ADDRESS_PROMPT_VERSION: PROMPT.replace(
        "and the complete printed merchant address.",
        "and the complete printed merchant address. For address, copy the entire merchant address block "
        "in printed line order, including building or mall, unit, street, city, and postal code when printed. "
        "Preserve house and lot numbers, street codes, postal codes, punctuation and spelling. "
        "Omit promotional text, phone numbers and addresses of other branches. Do not infer missing text.",
    ),
}
