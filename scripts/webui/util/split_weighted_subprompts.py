from util.imports import *

# grabs all text up to the first occurrence of ':' as sub-prompt
# takes the value following ':' as weight
# if ':' has no value defined, defaults to 1.0
# repeats until no text remaining
def split_weighted_subprompts(input_string, normalize=True):
	parsed_prompts = [(match.group("prompt"), float(match.group("weight") or 1)) for match in re.finditer(prompt_parser, input_string)]
	if not normalize:
		return parsed_prompts
	# this probably still doesn't handle negative weights very well
	weight_sum = sum(map(lambda x: x[1], parsed_prompts))
	return [(x[0], x[1] / weight_sum) for x in parsed_prompts]
