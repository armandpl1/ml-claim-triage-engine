"""Synthetic adjuster notes for claims.

freMTPL2 has no text, so each claim gets a short adjuster note generated from
its real features, and the note's category becomes the NLP model's label.

Notes are deterministic: the random source for a claim is seeded from its
claim_id (with a stable hash, not Python's per-process hash()), so the same
claim always gets the same note, in any order and on any machine.

How notes relate to the real data, so the NLP task is learnable but not
trivial:
    - category chances depend on the claim amount (bodily injury is far more
      likely on expensive claims, glass on cheap ones), vehicle age and
      population density
    - wording mentions damage severity, which correlates with the amount
    - a share of notes contain a realistic typo or abbreviation
"""

from __future__ import annotations

import hashlib
import random

CATEGORIES = ["collision", "theft", "glass", "weather", "bodily_injury"]

TEMPLATES = {
    "collision": [
        "Insured vehicle {impact} at {place}. {severity} damage to {part}.",
        "Rear-ended at {place} by third party. {severity} damage to {part}, vehicle {driveable}.",
        "Low visibility, insured hit a {obstacle} near {place}. {severity} {part} damage.",
    ],
    "theft": [
        "Vehicle reported stolen from {place} overnight. Police report filed.",
        "Break-in at {place}: {stolen} taken, {part} forced open.",
        "Insured found vehicle missing from {place}. Recovered {recovered}.",
    ],
    "glass": [
        "Stone chip on windscreen while driving on {road}. {severity} crack.",
        "Side window broken at {place}, no other damage reported.",
        "Windscreen cracked during {weather}. Replacement requested.",
    ],
    "weather": [
        "Hail storm damaged roof and {part} while parked at {place}.",
        "Vehicle flooded during {weather} at {place}. {severity} water damage.",
        "Tree branch fell on vehicle during {weather}. {severity} damage to {part}.",
    ],
    "bodily_injury": [
        "Collision at {place}, third party taken to hospital with {injury}.",
        "Pedestrian struck at {place}, reports {injury}. Liability under review.",
        "Insured and passenger injured at {place}: {injury}. Medical reports pending.",
    ],
}

SLOTS = {
    "impact": ["collided with another car", "skidded into a barrier", "hit a parked vehicle"],
    "place": [
        "a roundabout",
        "a supermarket car park",
        "a junction",
        "the motorway",
        "a city centre street",
        "a residential road",
    ],
    "part": ["front bumper", "rear door", "bonnet", "wing mirror", "side panel", "headlights"],
    "driveable": ["still driveable", "towed from scene", "not driveable"],
    "obstacle": ["post", "wall", "fence", "kerb"],
    "stolen": ["radio", "laptop bag", "spare wheel", "personal items"],
    "recovered": ["two days later with damage", "burnt out", "intact a week later"],
    "road": ["the motorway", "a country road", "a gravel road"],
    "weather": ["a storm", "heavy rain", "a hail storm", "a cold snap"],
    "injury": ["whiplash", "a broken arm", "back injuries", "minor cuts", "concussion"],
}

SEVERITY_WORDS = {
    "low": ["Minor", "Light", "Superficial"],
    "medium": ["Moderate", "Noticeable", "Significant"],
    "high": ["Severe", "Extensive", "Major"],
}

# Realistic adjuster shorthand and typos
TYPOS = [
    ("vehicle", "veh"),
    ("insured", "ins'd"),
    ("damage", "dmg"),
    ("reported", "reprted"),
    ("police", "polcie"),
]


def stable_seed(*parts: object) -> int:
    """A seed that is the same in every process and on every machine."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:16], 16)


def category_weights(amount: float, vehicle_age: int, density: int) -> list[float]:
    """Relative chance of each category, in CATEGORIES order."""
    weights = {
        "collision": 50.0,
        "theft": 10.0,
        "glass": 15.0,
        "weather": 10.0,
        "bodily_injury": 5.0,
    }
    if amount > 5000:
        weights["bodily_injury"] *= 8
        weights["glass"] *= 0.1
    elif amount < 800:
        weights["glass"] *= 3
        weights["bodily_injury"] *= 0.3
    if vehicle_age <= 3:
        weights["theft"] *= 1.8  # newer cars are stolen more
    if density > 5000:
        weights["theft"] *= 1.5  # dense urban areas
        weights["weather"] *= 0.6
    return [weights[c] for c in CATEGORIES]


def severity_band(amount: float) -> str:
    if amount < 1000:
        return "low"
    if amount < 5000:
        return "medium"
    return "high"


def generate_note(
    claim_id: str,
    amount: float,
    vehicle_age: int,
    density: int,
    seed: int,
    typo_rate: float = 0.08,
) -> tuple[str, str]:
    """Return (category, note text) for one claim. Deterministic per claim_id."""
    rng = random.Random(stable_seed(seed, claim_id))
    category = rng.choices(CATEGORIES, category_weights(amount, vehicle_age, density))[0]
    template = rng.choice(TEMPLATES[category])
    fills = {slot: rng.choice(options) for slot, options in sorted(SLOTS.items())}
    fills["severity"] = rng.choice(SEVERITY_WORDS[severity_band(amount)])
    text = template.format(**fills)
    if rng.random() < typo_rate:
        word, typo = rng.choice(TYPOS)
        text = text.replace(word, typo).replace(word.capitalize(), typo.capitalize())
    return category, text
