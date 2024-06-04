import re

RE_START_DASH = [r"()– "]

RE_NEWSPAPER = r"(?:[Tt]he )?(?:St\. )?(?:[A-Z][A-z]+[ -.]){0,3}(?:of )?(?:[A-Z][A-z]+)(?: \d{1,2})?(?:\'s)?(?:\.com|\.edu)?"

RE_NAME = r"[A-Z][a-z]+ [A-Z][a-z]+(?:-[A-Z][a-z]+)?"
RE_NAMES = r"{}(?: and {})?".format(RE_NAME, RE_NAME)

RE_REPORT = [
    r',("?) {} (?:reports|notes|finds|quips|writes|explains|said)(?=[\.,])'.format(
        RE_NEWSPAPER
    ),
    r',("?) (?:reports|notes|finds|quips|writes|explains) {}(?=[\.,])'.format(
        RE_NEWSPAPER
    ),
    r',("?) according to a (?:new )? report by {}(?=[\.,])'.format(RE_NEWSPAPER),
    r',("?) according to a (?:new )? (?:report|press release) by {}(?=[\.,])'.format(
        RE_NEWSPAPER
    ),
    r',("?) according to {}(?=[\.,])'.format(RE_NEWSPAPER),
    r',("?) per {}(?=[\.,])'.format(RE_NEWSPAPER),
    r",”( )(?:s)?he writes|explains in the {}, “".format(RE_NEWSPAPER),
]
RE_REMOVE_ALL = [re.compile(r) for r in RE_START_DASH + RE_REPORT]

BAD_LAST_SENT_START = [
    "Click here ",
    "Click for ",
    "Click through ",
    "Click to read ",
    "Read more ",
    "More details ",
    "Read the full ",
    "For more ",
    "(For more ",
    "Head to the ",
    "Read ",
]
BAD_LAST_SENT_END = [" here.", "here.)"]


def clean_sentence(sent):
    sent = re.sub(
        r',(["”]?) (?:{}) (?:reports|notes|finds|quips|writes|explains)( in the {})?\.$'.format(
            RE_NAMES, RE_NEWSPAPER
        ),
        r".\1",
        sent,
    )
    sent = re.sub(
        r',(["”]?) (?:{}) (?:reports|notes|finds|quips|writes|explains)\.$'.format(
            RE_NEWSPAPER
        ),
        r".\1",
        sent,
    )
    sent = re.sub(r"^{} reports on ".format(RE_NEWSPAPER), "Articles report on ", sent)
    sent = re.sub(
        r"^{} (?:reports|notes|finds|writes|explains) that ([A-z])".format(
            RE_NEWSPAPER
        ),
        lambda x: x.group(1).capitalize(),
        sent,
    )
    sent = re.sub(
        r"^{} (?:reports|notes|finds|writes|explains) ([A-z])".format(RE_NEWSPAPER),
        lambda x: x.group(1).capitalize(),
        sent,
    )
    sent = re.sub(
        r"^As {} (?:reports|notes|finds|writes|explains) ([a-z])".format(RE_NEWSPAPER),
        lambda x: x.group(1).capitalize(),
        sent,
    )
    sent = re.sub(
        r"^According to {}, ([a-z])".format(RE_NEWSPAPER),
        lambda x: x.group(1).capitalize(),
        sent,
    )
    sent = re.sub(
        r',("?) according to .*?(?=[\.,])', lambda x: x.group(1).capitalize(), sent
    )
    sent = re.sub(r',("?)[^,]*said(?=[\.,])', lambda x: x.group(1).capitalize(), sent)
    return sent


def remove_byline(text):
    regexes = [
        ".*[a-zA-Z\s|\,|\.|\(|\)\[\]]+?[\s]?[\u2013|\u2014|\-]+[\s]?(?![a-z|\s]+)",
        # dashes: e.g., 'ATLANTIC CITY, N.J. (AP) - ', must be followed by capital letters
        "^[A-Z|\,|\.|\(|\)\s]*[\u2013|\u2014|\-|\u003A|>\u2022]+[\s]?",
        # ALL CAPITAL followed by dash or colon or •: e.g., 'NEW DELHI-'
        "^.*[\(\[][A-Z]*[\)\]]\s(?![a-z|\s]+)",  # e.g., "Dehradun/New Delhi, Feb 7 (PTI) [followed by capital letters]"
        "^[a-zA-Z\s\,]*\u003A\s(?![a-z|\s]+)",  # e.g., "Washington: Suspected ", "[NFA] Captain Tom Moore"
    ]
    for r in regexes:
        text = re.sub(r, "", text)
    return text


def get_quotes(text):
    x = re.findall(r'"([^"]*)"|\'([^\']*)\'|“([^”]*)”', text)
    quotes = [ti for t in x for ti in t if ti]
    quotes = [q.replace(",", "") for q in quotes]
    quotes = [q.replace(".", "") for q in quotes]
    return quotes


def evaluate_quote_precision(summary, article):
    quotes = get_quotes(summary)
    if quotes:
        return sum([q in article for q in quotes]) / len(quotes)
    else:
        return 1


if __name__ == "__main__":
    clean_sentence(
        "Two FBI agents were shot dead and three were wounded on Tuesday while serving a search warrant on a suspect in a child pornography case in the southern US state of Florida, the FBI said."
    )
    clean_sentence(
        "Comerica Bank increased its position in iShares Russell 1000 Value ETF (NYSEARCA:IWD) by 53.6% during the fourth quarter, according to the company in its most recent filing with the Securities and Exchange Commission (SEC)."
    )
    clean_sentence(
        "U.S. President Joe Biden will announce a presidential memorandum on Thursday protecting the rights of lesbian, gay, bisexual, transgender and queer (LGBTQ) people worldwide, national security adviser Jake Sullivan said."
    )
