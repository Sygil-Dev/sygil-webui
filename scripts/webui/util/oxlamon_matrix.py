from util.imports import *

def oxlamon_matrix(prompt, seed, n_iter, batch_size):
	pattern = re.compile(r'(,\s){2,}')

	class PromptItem:
		def __init__(self, text, parts, item):
			self.text = text
			self.parts = parts
			if item:
				self.parts.append( item )

	def clean(txt):
		return re.sub(pattern, ', ', txt)

	def getrowcount( txt ):
		for data in re.finditer( ".*?\\((.*?)\\).*", txt ):
			if data:
				return len(data.group(1).split("|"))
			break
		return None

	def repliter( txt ):
		for data in re.finditer( ".*?\\((.*?)\\).*", txt ):
			if data:
				r = data.span(1)
				for item in data.group(1).split("|"):
					yield (clean(txt[:r[0]-1] + item.strip() + txt[r[1]+1:]), item.strip())
			break

	def iterlist( items ):
		outitems = []
		for item in items:
			for newitem, newpart in repliter(item.text):
				outitems.append( PromptItem(newitem, item.parts.copy(), newpart) )

		return outitems

	def getmatrix( prompt ):
		dataitems = [ PromptItem( prompt[1:].strip(), [], None ) ]
		while True:
			newdataitems = iterlist( dataitems )
			if len( newdataitems ) == 0:
				return dataitems
			dataitems = newdataitems

	def classToArrays( items, seed, n_iter ):
		texts = []
		parts = []
		seeds = []

		for item in items:
			itemseed = seed
			for i in range(n_iter):
				texts.append( item.text )
				parts.append( f"Seed: {itemseed}\n" + "\n".join(item.parts) )
				seeds.append( itemseed )
				itemseed += 1                

		return seeds, texts, parts

	all_seeds, all_prompts, prompt_matrix_parts = classToArrays(getmatrix( prompt ), seed, n_iter)
	n_iter = math.ceil(len(all_prompts) / batch_size)

	needrows = getrowcount(prompt)
	if needrows:
		xrows = math.sqrt(len(all_prompts))
		xrows = round(xrows)
		# if columns is to much
		cols = math.ceil(len(all_prompts) / xrows)
		if cols > needrows*4:
			needrows *= 2

	return all_seeds, n_iter, prompt_matrix_parts, all_prompts, needrows
