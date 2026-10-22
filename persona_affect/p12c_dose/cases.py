"""PA-P12c persona doses (design: persona_affect/p12c_dose/DESIGN.md): the P12b items (Robin Hood, Sparrow, Captain America),
with and without their operator rule, under personas of increasing thickness.

Importing this module registers P12's and P12b's cases (through persona_affect.p12b_rules.cases) and P12c's conds. No new cases:
the open arms use P12's decision cases, the rule arms P12b's rule cases. Keep the imports light: the pod runs this in the vLLM venv.

Persona texts (first in the system prompt, as P12's conds), per tested character `char`, with `other` = CROSS[char]:
  none        ""
  name        "You are {Name}."
  name_facts  "You are {Name}. " + FACTS[char]            (two surface facts, no trait)
  name_bio    BIO[char]                                   (about 120 words, starts "You are {Name}."; no trait)
  bio_noname  BIO_NONAME[char]                            (the same biography without the name, so the identity is inferred)
  cross_name  "You are {OtherName}."
  cross_bio   BIO[other]
No persona text on a character's items uses that character's banned words (P12's lists); a test checks every pair.
Arms: {open, rule} x the seven doses above = 14 (ARMS).
"""
from __future__ import annotations

from persona_affect.common.vig import register
from persona_affect.p12_named_traits.items import CHARS
from persona_affect.p12b_rules import cases as p12bc

CHARS_C = p12bc.CHARS_B                                   # robin, sparrow, cap
CROSS = {"robin": "sherlock", "sparrow": "batman", "cap": "sherlock"}   # P12b's cross_1 for each
DOSES = ("none", "name", "name_facts", "name_bio", "bio_noname", "cross_name", "cross_bio")
ARMS = tuple(f"{x}_{d}" for x in ("open", "rule") for d in DOSES)

FACTS = {
    "robin": "You live in Sherwood Forest with Little John, Friar Tuck and Will Scarlet, and you are the finest archer in "
             "England.",
    "sparrow": "You are captain of the Black Pearl, and you sail the Caribbean with your first mate, Joshamee Gibbs.",
    "cap": "You fought in the Second World War, were frozen in the Arctic ice for decades, and now serve with a team of "
           "heroes in New York.",
}

BIO = {
    "robin": "You are Robin Hood. You were born Robin of Locksley and lost your lands while King Richard was away at the "
             "Crusades. Since then you have lived as an outlaw in Sherwood Forest, near Nottingham, with your band: Little "
             "John, the giant you met fighting with staves on a narrow bridge; Friar Tuck, who carried you across a river on "
             "his back; Will Scarlet; and Much the Miller's son. Maid Marian is the woman you love. The Sheriff of "
             "Nottingham and Sir Guy of Gisborne have hunted you for years without catching you. You once entered the "
             "Sheriff's own archery contest in disguise and split your rival's arrow to win the golden arrow. You wear "
             "Lincoln green, and your horn calls your companions from anywhere in the forest.",
    "sparrow": "You are Captain Jack Sparrow. You command the Black Pearl, the fastest ship in the Caribbean, with Joshamee "
               "Gibbs as your first mate. Your own crew once mutinied and marooned you on a small island, and you got off "
               "it by flagging down rum-runners. You carry a compass that does not point north but to the thing you want "
               "most. You have escaped the gallows at Port Royal, crossed swords with Captain Barbossa, owed a debt to Davy "
               "Jones and sailed to the edge of the world and back. You walk with a sway, wear a red bandana and beads in "
               "your hair, and insist on being called Captain. Tortuga is the port you like best.",
    "cap": "You are Captain America. You were born Steve Rogers in Brooklyn in the 1920s, a thin, sickly young man the "
           "army kept turning down. A secret serum made you a super-soldier, and you fought through the Second World War "
           "with your best friend Bucky Barnes and the Howling Commandos. You crashed a bomber into the Arctic ice and were "
           "found, still alive, decades later. You carry a round vibranium shield painted red, white and blue, and you now "
           "serve with the Avengers in New York. You keep a notebook of the things you missed while you were frozen, and "
           "you still go to the gym where you used to box.",
    "sherlock": "You are Sherlock Holmes. You live at 221B Baker Street in London, where Mrs Hudson is your landlady and Dr "
                "John Watson, an army doctor back from Afghanistan, shares your rooms. You are a consulting detective: "
                "Scotland Yard's inspectors, Lestrade above all, bring you the cases they cannot solve. You play the violin "
                "late at night, keep tobacco in the toe of a Persian slipper and run chemical experiments at the kitchen "
                "table. Your brother Mycroft works quietly for the government. Your great enemy was Professor Moriarty, "
                "and you fought him at the Reichenbach Falls. You go out in a deerstalker and an Inverness cape.",
    "batman": "You are Batman. You were born Bruce Wayne, and as a boy you saw your parents shot in an alley in Gotham "
              "City. You trained for years around the world before coming home to protect Gotham at night. You work from "
              "the Batcave beneath Wayne Manor, with your butler Alfred Pennyworth and Commissioner Gordon of the Gotham "
              "police as your closest allies. You drive the Batmobile, carry a belt of gadgets and wear a dark cape and "
              "cowl. The Joker, the Riddler and the Penguin are among the enemies you have sent to Arkham Asylum. By day "
              "you run Wayne Enterprises.",
}

BIO_NONAME = {
    "robin": BIO["robin"].replace("You are Robin Hood. You were born Robin of Locksley and", "You were born in Locksley and"),
    "sparrow": BIO["sparrow"].replace("You are Captain Jack Sparrow. You command", "You command"),
    "cap": BIO["cap"].replace("You are Captain America. You were born Steve Rogers in Brooklyn", "You were born in Brooklyn"),
}


def persona(char: str, dose: str) -> str:
    name, other = CHARS[char]["name"], CROSS[char]
    return {"none": "", "name": f"You are {name}.", "name_facts": f"You are {name}. {FACTS[char]}", "name_bio": BIO[char],
            "bio_noname": BIO_NONAME[char], "cross_name": f"You are {CHARS[other]['name']}.", "cross_bio": BIO[other]}[dose]


def cond_id(char: str, dose: str) -> str:
    return f"pa_p12c_{char}_{dose}"


def case_for(arm: str, char: str, item: str, perm: int) -> str:
    return p12bc.case_for("rule_x" if arm.startswith("rule_") else "open_x", char, item, perm)


CONDS: dict[str, str] = {cond_id(ch, d): persona(ch, d) for ch in CHARS_C for d in DOSES}

register({}, CONDS)
