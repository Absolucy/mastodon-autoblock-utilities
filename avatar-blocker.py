#!/usr/bin/env python3
import requests
from argparse import ArgumentParser
from cachetools import LRUCache, TTLCache, cached, keys
from colorlog import ColoredFormatter, StreamHandler
from configparser import ConfigParser
from io import BytesIO
from logging import DEBUG, INFO, getLogger
from mastodon import CallbackStreamListener, Mastodon
from PIL import Image
from signal import SIGINT, SIGTERM, signal
from sys import exit
from time import sleep
from transformers import pipeline

logger = getLogger("avatar-blocker")
logger.setLevel(INFO)
handler = StreamHandler()
handler.setFormatter(ColoredFormatter("%(asctime)s %(log_color)s%(levelname)s%(reset)s %(name)s: %(message)s"))
logger.addHandler(handler)

parser = ArgumentParser()
parser.add_argument("--model", "-m", type=str, help="what model to use for avatar classification (default: google/vit-base-patch16-224)")
parser.add_argument(
    "--instance",
    "-i",
    type=str,
    help="the Mastodon instance you reside on. don't include the http(s) portion, just the base url (default: mastodon.social)"
)
parser.add_argument("--access-token", "-t", type=str, help="your mastodon access token.")
parser.add_argument(
    "--auto-block",
    "-a",
    help="whether to automatically block, or just leave it to the user to block (default: will not automatically block)",
    action="store_true"
)
parser.add_argument(
    "--bad-categories", "-b", type=str, help="which classifications are considered 'bad', separated by a comma (default: \"bad\")"
)
parser.add_argument(
    "--minimum-score", "-s", type=float, help="the minimum score for a bad classification to be considered (default: 0.75)"
)
parser.add_argument("--watch-hashtags", "-wh", type=str, help="which hashtags to watch closely, separated by a comma")
parser.add_argument(
    "--include-following",
    "-if",
    help="include people you are following into consideration for judgement (default: users you follow are excluded)",
    action="store_true"
)
parser.add_argument(
    "--exclude-followers",
    "-ef",
    help="exclude people following you from consideration for judgement (default: users following you are included)",
    action="store_true"
)
parser.add_argument(
    "--no-listen-public",
    "-np",
    help="disables the public stream listener",
    action="store_true"
)
parser.add_argument(
    "--no-listen-user",
    "-nu",
    help="disables the user stream listener",
    action="store_true"
)
parser.add_argument(
    "--image-cache-ttl", "-it", type=int, help="how long (in minutes) to cache images for (default: 45 minutes)"
)
parser.add_argument(
    "--relationship-cache-ttl",
    "-rt",
    type=int,
    help="how long (in minutes) to cache user relationships for (default: 6 hours)"
)
parser.add_argument(
    "--user-agent", "-ua", type=str, help="the User-Agent header to send to the Mastodon api (default: \"mastodon-autoblock-utilities/avatar-blocker/1.0\")"
)
parser.add_argument("--debug", "-d", help="debug logging", action="store_true")
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
bad_categories = (args.bad_categories or str(config.get("bad-categories", "bad"))).split(",")
minimum_score = args.minimum_score or float(config.get("minimum-score", 0.75))
watch_hashtags = (args.watch_hashtags or config.get("watch-hashtags", "")).split(",")
include_following = args.include_following or bool(config.get("include-following", False))
exclude_followers = args.exclude_followers or bool(config.get("exclude-followers", False))
no_listen_public = args.no_listen_public or bool(config.get("no-listen-public", False))
no_listen_user = args.no_listen_user or bool(config.get("no-listen-user", False))
image_cache_ttl = args.image_cache_ttl or int(config.get("image-cache-ttl", 45))
relationship_cache_ttl = args.relationship_cache_ttl or int(config.get("relationship-cache-ttl", 360))
user_agent = args.user_agent or config.get("user-agent", "mastodon-autoblock-utilities/avatar-blocker/1.0")

headers = {"User-Agent": user_agent}

if args.debug:
	logger.setLevel(DEBUG)

logger.info("Loading model '%s' from HuggingFace", model)
classifier = pipeline("image-classification", model=model)


def cache_key_acct(*args, **kwargs):
	global logger
	if len(args) != 1:
		logger.warn("encountered an invalid account function while caching: args=%s", args)
		return keys.hashkey(*args, **kwargs)
	account = args[0]
	if not "id" in account:
		logger.warn("encountered an invalid account while caching: account=%s", account)
		return keys.hashkey(*args, **kwargs)
	return hash(account["id"])


@cached(cache=TTLCache(maxsize=1024, ttl=image_cache_ttl * 60), key=cache_key_acct)
def download_pfp(account):
	global logger, headers
	if "avatar_static" not in account:
		return None
	name = account["acct"]
	pfp_url = account["avatar_static"]
	response = requests.get(pfp_url, headers=headers, timeout=10)
	if response.ok:
		try:
			return Image.open(BytesIO(response.content)).convert("RGB").resize((224, 224))
		except:
			logger.exception("failed to get avatar for %s", name)
			return None
	else:
		logger.error("failed to download avatar for %s: http code %i", name, response.status_code)
		return None


@cached(cache=TTLCache(maxsize=1024, ttl=relationship_cache_ttl * 60), key=cache_key_acct)
def get_relationship(account):
	global logger, mastodon
	id = account["id"]
	name = account["acct"]
	try:
		return mastodon.account_relationships(id)
	except:
		logger.exception("failed to get relationships with %s", name)
		return


@cached(cache=LRUCache(maxsize=512), key=cache_key_acct)
def is_account_bad(account):
	global logger, classifier, bad_categories, minimum_score
	if not "avatar_static" in account:
		return False
	user = account["acct"]
	if account["avatar_static"].endswith("/missing.png"):
		logger.debug("user @%s has default pfp, skipping", user)
		return
	logger.debug("checking user %s", user)
	pfp = download_pfp(account)
	if not pfp:
		return False
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
	global logger, mastodon, auto_block
	if "account" in data:
		account = data["account"]
		if "id" not in account:
			return
		if is_account_bad(account):
			id = account["id"]
			name = account["acct"]
			relationships = get_relationship(account) or {}
			if isinstance(relationships, list):
				if len(relationships) > 0:
					relationships = relationships[0]
				else:
					relationships = {}
			if not include_following and relationships.get("following", False):
				logger.info("%s would be bad, but we're following them, so they get a pass", name)
				return
			if exclude_followers and relationships.get("followed_by", False):
				logger.info("%s would be bad, but they're following us, so they get a pass", name)
				return
			logger.info("oh no, %s is bad!", name)
			if auto_block:
				logger.info("blocking %s", name)
				try:
					mastodon.account_block(id)
				except:
					logger.exception("failed to block %s", name)


def signal_handler(sig, frame):
	global logger
	logger.warning("Ctrl+C pressed, exiting")
	exit(0)

def check_hashtag_timeline(hashtag):
	global mastodon, logger
	logger.info("Checking timeline of #%s", hashtag)
	try:
		timeline = mastodon.timeline_hashtag(hashtag)
	except:
		logger.exception("failed to get timeline of #%s", hashtag)
		return
	for status in timeline:
		 on_stream(status)

try:
	mastodon = Mastodon(access_token=access_token, api_base_url=f"https://{instance}")
	me = mastodon.me()
except:
	logger.exception("failed to log into mastodon")
logger.info("logged into Mastodon as @%s", me.acct)

if not no_listen_user:
	mastodon.stream_user(CallbackStreamListener(update_handler=on_stream), run_async=True, reconnect_async=True)
	logger.info("listening to user stream")

if not no_listen_public:
	mastodon.stream_public(CallbackStreamListener(update_handler=on_stream), run_async=True, reconnect_async=True)
	logger.info("listening to public stream")

for hashtag in watch_hashtags:
	hashtag = hashtag.removeprefix("#").strip()
	mastodon.stream_hashtag(
	    hashtag, CallbackStreamListener(update_handler=on_stream), run_async=True, reconnect_async=True
	)
	logger.info("listening to #%s stream", hashtag)
	check_hashtag_timeline(hashtag)

signal(SIGINT, signal_handler)
signal(SIGTERM, signal_handler)
while True:
	sleep(1)
