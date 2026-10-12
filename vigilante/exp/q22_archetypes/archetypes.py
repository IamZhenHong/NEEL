"""Item 22 personas: 10 behaviour-derived assistant traits x 3 roles (famous and generic mixed), 10 non-assistant
characters as a floor/contrast, and two assistant baselines. System prompt = "You are {role}." (name only, like P1)."""

TRAITS = {
    "service": ["a hotel concierge", "Jeeves, the valet from the P. G. Wodehouse stories", "a customer service representative"],
    "deference": ["a butler", "Alfred Pennyworth, Batman's butler", "a genie bound to grant wishes"],
    "teaching": ["a patient primary-school teacher", "Mr. Rogers, the children's TV host", "a private tutor"],
    "care": ["a hospice nurse", "Mary Poppins", "a counsellor"],
    "honesty": ["a reference librarian", "Data, the android officer from Star Trek", "a newspaper fact-checker"],
    "caution": ["a workplace safety officer", "a pharmacist", "a risk-averse lawyer"],
    "balance": ["a mediator", "a BBC news presenter", "a diplomat"],
    "humility": ["a careful research scientist", "Socrates", "a junior analyst"],
    "expertise": ["a senior software engineer", "Sherlock Holmes", "a management consultant"],
    "cheer": ["a cheerful flight attendant", "Ted Lasso", "a children's party entertainer"],
}
OTHERS = ["a pirate captain", "a medieval knight", "a grumpy cat", "a sports commentator", "a drill sergeant",
          "a conspiracy theorist", "a cowboy", "a Shakespearean actor", "a hard-boiled noir detective", "a surfer"]

PERSONAS = {"default": None, "helpful_assistant": "You are a helpful assistant."}
for t, roles in TRAITS.items():
    for i, r in enumerate(roles):
        PERSONAS[f"{t}_{i}"] = f"You are {r}."
for i, r in enumerate(OTHERS):
    PERSONAS[f"other_{i}"] = f"You are {r}."
TRAIT_OF = {k: k.rsplit("_", 1)[0] for k in PERSONAS if k not in ("default", "helpful_assistant") and not k.startswith("other_")}
