# The horde url
horde_url = "https://stablehorde.net"
# Give a cool name to your instance
horde_name = "My Awesome Instance"
# The api_key identifies a unique user in the horde
# Visit https://stablehorde.net/register to create one before you can join
horde_api_key = "0000000000"
# Put other users whose prompts you want to prioritize.
# The owner's username is always included so you don't need to add it here, unless you want it to have lower priority than another user
horde_priority_usernames = []
# The amount of power your system can handle
# 8 means 512*512. Each increase increases the possible resoluion by 64 pixes
# So if you put this to 2 (the minimum, your SD can only generate 64x64 pixels
# If you put this to 32, it is equivalent to 1024x1024 pixels
horde_max_power = 8
# Set this to false, if you do not want your worker to receive requests for NSFW generations
horde_nsfw = True
# A list of words which you do not want to your worker to accept
horde_blacklist = []
# A list of words for which you always want to allow the NSFW censor filter, even when this worker is in NSFW mode
horde_censorlist = []
