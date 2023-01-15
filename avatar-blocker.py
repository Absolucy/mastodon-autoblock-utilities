#!/usr/bin/env python3
from argparse import ArgumentParser
from configparser import ConfigParser
from logging import getLogger, INFO, DEBUG
from transformers import pipeline
from colorlog import StreamHandler, ColoredFormatter
from mastodon import Mastodon, CallbackStreamListener
from PIL import Image
from io import BytesIO
import requests

headers = {"User-Agent": "mastodon-autoblock-utilities/avatar-blocker/1.0"}

logger = getLogger("avatar-blocker")
logger.setLevel(INFO)
handler = StreamHandler()
handler.setFormatter(
    ColoredFormatter(
        "%(asctime)s %(log_color)s%(levelname)s%(reset)s %(name)s: %(message)s"
    ))
logger.addHandler(handler)

parser = ArgumentParser()
parser.add_argument("--model",
                    "-m",
                    type=str,
                    help="what model to use for avatar classification")
parser.add_argument(
    "--instance",
    "-i",
    type=str,
    help=
    "the mastodon instance you reside on. don't include the http(s) portion, just the base url."
)
parser.add_argument("--access-token",
                    "-t",
                    type=str,
                    help="your mastodon access token.")
parser.add_argument(
    "--auto-block",
    "-a",
    type=bool,
    help=
    "whether to automatically block, or just leave it to the user to block",
    action="store_true")
parser.add_argument(
    "--bad-categories",
    "-b",
    type=str,
    help="which classifications are considered 'bad', separated by a comma")
parser.add_argument(
    "--minimum-score",
    "-s",
    type=float,
    help="the minimum score for a bad classification to be considered")
parser.add_argument("--debug",
                    "-d",
                    type=bool,
                    help="debug logging",
                    action="store_true")
args = parser.parse_args()

config = ConfigParser()
config.read("avatar.ini")
if "config" in config:
	config = config["config"]
else:
	logger.warn("config file not setup, I hope you have defaults set!")
	config = {}

instance = args.instance or str(config.get("instance", "mastodon.social"))
access_token = args.access_token or str(config.get("access-token", "INVALID"))
model = args.model or str(config.get("model", "google/vit-base-patch16-224"))
auto_block = args.auto_block or bool(config.get("auto-block", False))
bad_categories = (args.bad_categories
                  or str(config.get("bad-categories", "bad"))).split(",")
minimum_score = args.minimum_score or float(config.get("minimum-score", 0.75))

if args.debug:
	logger.setLevel(DEBUG)

logger.info("Loading model '%s' from HuggingFace", model)
classifier = pipeline("image-classification", model=model)


def download_pfp(url, username):
	global logger, headers
	response = requests.get(url, headers=headers, timeout=10)
	if response.ok:
		try:
			return Image.open(BytesIO(response.content)).resize((224, 224))
		except:
			logger.exception("failed to get avatar for %s", username)
			return None
	else:
		logger.error("failed to download avatar for %s: http code %i",
		             username, response.status_code)
		return None


def is_account_bad(account):
	global logger, classifier, bad_categories, minimum_score
	if not "avatar_static" in account:
		return
	pfp_url = account["avatar_static"]
	user = account["acct"]
	if pfp_url.endswith("/missing.png"):
		logger.debug("user @%s has default pfp, skipping", user)
		return
	logger.debug("checking user %s", user)
	pfp = download_pfp(pfp_url, user)
	if not pfp:
		return
	try:
		classification = classifier(pfp)
		logger.debug("classification of %s: %s", user, classification)
		for cls in classification:
			if not "score" in cls or not "label" in cls:
				continue
			score = cls["score"]
			label = cls["label"]
			if score < minimum_score:
				continue
			if label in bad_categories:
				return True
	except:
		logger.exception("failed to classify pfp of %s", user)
	return False


def on_stream(data):
	global logger
	if "account" in data:
		if is_account_bad(data["account"]):
			logger.info("oh no")


try:
	mastodon = Mastodon(access_token=access_token,
	                    api_base_url=f"https://{instance}")
	me = mastodon.me()
except:
	logger.exception("failed to log into mastodon")
logger.info("Logged into Mastodon as @%s", me.acct)
mastodon.stream_user(CallbackStreamListener(update_handler=on_stream))
