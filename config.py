import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

LOOKBACK_DAYS = 14
REQUEST_TIMEOUT = 15

USER_AGENT = (
    "Mozilla/5.0 "
    "(Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 "
    "(KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)

FED_BASE_URL = "https://www.federalreserve.gov"
FED_SPEECHES_URL = "https://www.federalreserve.gov/newsevents/speeches.htm"
FED_TESTIMONY_URL = "https://www.federalreserve.gov/newsevents/testimony.htm"
FED_PRESS_RELEASES_RSS = "https://www.federalreserve.gov/feeds/press_all.xml"


# 2026-08 current roster.
# voter=True means 2026 FOMC voting member.
FED_MEMBER_INFO = {
    # Board of Governors
    "Kevin Warsh": {
        "role_ko": "연준 의장",
        "role_en": "Chair, Board of Governors",
        "fed": "Board of Governors",
        "voter": True,
    },
    "Philip Jefferson": {
        "role_ko": "부의장",
        "role_en": "Vice Chair, Board of Governors",
        "fed": "Board of Governors",
        "voter": True,
    },
    "Michelle Bowman": {
        "role_ko": "감독 담당 부의장",
        "role_en": "Vice Chair for Supervision, Board of Governors",
        "fed": "Board of Governors",
        "voter": True,
    },
    "Michael Barr": {
        "role_ko": "연준 이사",
        "role_en": "Member, Board of Governors",
        "fed": "Board of Governors",
        "voter": True,
    },
    "Lisa Cook": {
        "role_ko": "연준 이사",
        "role_en": "Member, Board of Governors",
        "fed": "Board of Governors",
        "voter": True,
    },
    "Jerome Powell": {
        "role_ko": "연준 이사",
        "role_en": "Member, Board of Governors",
        "fed": "Board of Governors",
        "voter": True,
    },
    "Christopher Waller": {
        "role_ko": "연준 이사",
        "role_en": "Member, Board of Governors",
        "fed": "Board of Governors",
        "voter": True,
    },

    # Regional Federal Reserve Banks
    "Susan Collins": {
        "role_ko": "보스턴 연은 총재",
        "role_en": "President, Federal Reserve Bank of Boston",
        "fed": "Boston Fed",
        "voter": False,
    },
    "John Williams": {
        "role_ko": "뉴욕 연은 총재",
        "role_en": "President, Federal Reserve Bank of New York",
        "fed": "New York Fed",
        "voter": True,
    },
    "Anna Paulson": {
        "role_ko": "필라델피아 연은 총재",
        "role_en": "President, Federal Reserve Bank of Philadelphia",
        "fed": "Philadelphia Fed",
        "voter": True,
    },
    "Beth Hammack": {
        "role_ko": "클리블랜드 연은 총재",
        "role_en": "President, Federal Reserve Bank of Cleveland",
        "fed": "Cleveland Fed",
        "voter": True,
    },
    "Thomas Barkin": {
        "role_ko": "리치먼드 연은 총재",
        "role_en": "President, Federal Reserve Bank of Richmond",
        "fed": "Richmond Fed",
        "voter": False,
    },
    "Cheryl Venable": {
        "role_ko": "애틀랜타 연은 임시 총재",
        "role_en": "Interim President and CEO, Federal Reserve Bank of Atlanta",
        "fed": "Atlanta Fed",
        "voter": False,
    },
    "Austan Goolsbee": {
        "role_ko": "시카고 연은 총재",
        "role_en": "President, Federal Reserve Bank of Chicago",
        "fed": "Chicago Fed",
        "voter": False,
    },
    "Alberto Musalem": {
        "role_ko": "세인트루이스 연은 총재",
        "role_en": "President, Federal Reserve Bank of St. Louis",
        "fed": "St. Louis Fed",
        "voter": False,
    },
    "Neel Kashkari": {
        "role_ko": "미니애폴리스 연은 총재",
        "role_en": "President, Federal Reserve Bank of Minneapolis",
        "fed": "Minneapolis Fed",
        "voter": True,
    },
    "Jeffrey Schmid": {
        "role_ko": "캔자스시티 연은 총재",
        "role_en": "President, Federal Reserve Bank of Kansas City",
        "fed": "Kansas City Fed",
        "voter": False,
    },
    "Lorie Logan": {
        "role_ko": "댈러스 연은 총재",
        "role_en": "President, Federal Reserve Bank of Dallas",
        "fed": "Dallas Fed",
        "voter": True,
    },
    "Mary Daly": {
        "role_ko": "샌프란시스코 연은 총재",
        "role_en": "President, Federal Reserve Bank of San Francisco",
        "fed": "San Francisco Fed",
        "voter": False,
    },
}

FED_MEMBERS = list(FED_MEMBER_INFO)

MIN_SEGMENT_LENGTH = 100
MAX_SEGMENT_LENGTH = 4000

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-5-mini"
LLM_TEMPERATURE = 0

STANCE_HAWKISH = "HAWKISH"
STANCE_DOVISH = "DOVISH"
STANCE_NEUTRAL = "NEUTRAL"
STANCE_IRRELEVANT = "IRRELEVANT"

CONTENT_PRESCRIPTIVE = "PRESCRIPTIVE"
CONTENT_DESCRIPTIVE = "DESCRIPTIVE"
CONTENT_MIXED = "MIXED"
CONTENT_IRRELEVANT = "IRRELEVANT"

MIN_STANCE_SCORE = -1.0
MAX_STANCE_SCORE = 1.0
HAWKISH_THRESHOLD = 0.20
DOVISH_THRESHOLD = -0.20
VALIDATION_GAP_THRESHOLD = 0.40
