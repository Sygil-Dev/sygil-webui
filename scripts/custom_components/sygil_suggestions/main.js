
// parent document
var parentDoc = window.parent.document;
// iframe element in parent document
var frame = window.frameElement;
// the area to put the suggestions in
var suggestionArea = document.getElementById('suggestion_area');
var scrollArea = document.getElementById('scroll_area');
// button height is read when the first button gets created
var buttonHeight = -1;
// the maximum size of the iframe in buttons (3 x buttons height)
var maxHeightInButtons = 3;
// the prompt field connected to this iframe
var promptField = null;
// the category of suggestions
var activeCategory = contextCategory;

var conditionalButtons = null;

var contextCategory = "[context]";

var frameHeight = "calc( 3em - 3px + {} )";

var filterGroups = {nsfw_mild: "nsfw_mild", nsfw_basic: "nsfw_basic", nsfw_strict: "nsfw_strict", gore_mild: "gore_mild", gore_basic: "gore_basic", gore_strict: "gore_strict"};
var activeFilters = [filterGroups.nsfw_mild, filterGroups.nsfw_basic, filterGroups.gore_mild];

var triggers = {empty: "empty", nsfw: "nsfw", nude: "nude"};
var activeContext = [];

var triggerIndex = {};

var wordMap = {};
var tagMap = {};

// could pass in an array of specific stylesheets for optimization
function getAllCSSVariableNames(styleSheets = parentDoc.styleSheets){
   var cssVars = [];
   // loop each stylesheet
   for(var i = 0; i < styleSheets.length; i++){
      // loop stylesheet's cssRules
      try{ // try/catch used because 'hasOwnProperty' doesn't work
         for( var j = 0; j < styleSheets[i].cssRules.length; j++){
            try{
				//console.log(styleSheets[i].cssRules[j].selectorText);
               // loop stylesheet's cssRules' style (property names)
               for(var k = 0; k < styleSheets[i].cssRules[j].style.length; k++){
                  let name = styleSheets[i].cssRules[j].style[k];
                  // test name for css variable signiture and uniqueness
                  if(name.startsWith('--') && cssVars.indexOf(name) == -1){
                     cssVars.push(name);
                  }
               }
            } catch (error) {}
         }
      } catch (error) {}
   }
   return cssVars;
}

function currentFrameAbsolutePosition() {
  let currentWindow = window;
  let currentParentWindow;
  let positions = [];
  let rect;

  while (currentWindow !== window.top) {
    currentParentWindow = currentWindow.parent;
    for (let idx = 0; idx < currentParentWindow.frames.length; idx++)
      if (currentParentWindow.frames[idx] === currentWindow) {
        for (let frameElement of currentParentWindow.document.getElementsByTagName('iframe')) {
          if (frameElement.contentWindow === currentWindow) {
            rect = frameElement.getBoundingClientRect();
            positions.push({x: rect.x, y: rect.y});
          }
        }
        currentWindow = currentParentWindow;
        break;
      }
  }
  return positions.reduce((accumulator, currentValue) => {
    return {
      x: accumulator.x + currentValue.x,
      y: accumulator.y + currentValue.y
    };
  }, { x: 0, y: 0 });
}

// check if element is visible
function isVisible(e) {
    return !!( e.offsetWidth || e.offsetHeight || e.getClientRects().length );
}

// remove everything from the suggestion area
function ClearSuggestionArea(text = "")
{
	suggestionArea.innerHTML = text;
	conditionalButtons = [];
}

// update iframe size depending on button rows
function UpdateSize()
{
	// calculate maximum height
	var maxHeight = buttonHeight * maxHeightInButtons;
	
	var height = suggestionArea.lastChild.offsetTop + buttonHeight;
	// apply height to iframe
	frame.style.height = frameHeight.replace("{}", Math.min(height,maxHeight)+"px");
	scrollArea.style.height = frame.style.height;
}

// add a button to the suggestion area
function AddButton(label, action, dataTooltip = null, tooltipImage = null, pattern = null, data = null)
{
	// create span
	var button = document.createElement("span");
	// label it
	button.innerHTML = label;
	if(data != null)
	{
		// add category attribute to button, will be read on click
		button.setAttribute("data",data);
	}
	if(pattern != null)
	{
		// add category attribute to button, will be read on click
		button.setAttribute("pattern",pattern);
	}
	if(dataTooltip != null)
	{
		// add category attribute to button, will be read on click
		button.setAttribute("tooltip-text",dataTooltip);
	}
	if(tooltipImage != null)
	{
		// add category attribute to button, will be read on click
		button.setAttribute("tooltip-image",tooltipImage);
	}
	// add button function
	button.addEventListener('click', action, false);
	button.addEventListener('mouseover', ButtonHoverEnter);
	button.addEventListener('mouseout', ButtonHoverExit);
	// add button to suggestion area
	suggestionArea.appendChild(button);
	// get buttonHeight if not set
	if(buttonHeight < 0)
		buttonHeight = button.offsetHeight;
	return button;
}

// find visible prompt field to connect to this iframe
function GetPromptField()
{
	// get all prompt fields, the %% placeholder %% is set in python
	var all = parentDoc.querySelectorAll('textarea[placeholder="'+placeholder+'"]');
	// filter visible
	for(var i = 0; i < all.length; i++)
	{
		if(isVisible(all[i]))
		{
			promptField = all[i];
			promptField.addEventListener('input', OnChange, false);
			promptField.addEventListener('click', OnClick, false);
			promptField.addEventListener('keyup', OnKey, false);
			break;
		}
	}
}

function OnChange(e)
{
	ButtonConditions();
	ButtonUpdateContext(true);
}

function OnClick(e)
{
	ButtonUpdateContext(true);
}

function OnKey(e)
{
	if (e.keyCode == '37' || e.keyCode == '38' || e.keyCode == '39' || e.keyCode == '40') {
		ButtonUpdateContext(false);
	}
}

function getCaretPosition(ctrl) {
    // IE < 9 Support 
    if (document.selection) {
        ctrl.focus();
        var range = document.selection.createRange();
        var rangelen = range.text.length;
        range.moveStart('character', -ctrl.value.length);
        var start = range.text.length - rangelen;
        return {
            'start': start,
            'end': start + rangelen
        };
    } // IE >=9 and other browsers
    else if (ctrl.selectionStart || ctrl.selectionStart == '0') {
        return {
            'start': ctrl.selectionStart,
            'end': ctrl.selectionEnd
        };
    } else {
        return {
            'start': 0,
            'end': 0
        };
    }
}

function setCaretPosition(ctrl, start, end) {
    // IE >= 9 and other browsers
    if (ctrl.setSelectionRange) {
        ctrl.focus();
        ctrl.setSelectionRange(start, end);
    }
    // IE < 9 
    else if (ctrl.createTextRange) {
        var range = ctrl.createTextRange();
        range.collapse(true);
        range.moveEnd('character', end);
        range.moveStart('character', start);
        range.select();
    }
}

function isEmptyOrSpaces(str){
    return str === null || str.match(/^ *$/) !== null;
}

function ButtonUpdateContext(changeCategory)
{
	let targetCategory = contextCategory;
	let text = promptField.value;
	if(document.activeElement === promptField)
	{
		var pos = getCaretPosition(promptField).end;
		text = promptField.value.slice(0, pos);
	}
	
	activeContext = [];
	
	var parts = text.split(/[\.?!,]/);
	if(activeCategory == "Artists" && !isEmptyOrSpaces(parts[parts.length-1]))
	{
		return;
	}
	if(text == "")
	{
		activeContext.push(triggers.empty);
	}
	if(text.endsWith("by"))
	{
		changeCategory = true;
		targetCategory = "Artists";
		activeContext.push("Artists");
	}
	else
	{
		var parts = text.split(/[\.,!?;]/);
		parts = parts.reverse();
		
		parts.forEach( part =>
		{
			var words = part.split(" ");
			words = words.reverse();
			words.forEach( word =>
			{
				word = word.replace(/[^a-zA-Z0-9 \._\-]/g, '').trim().toLowerCase();
				word = WordToKey(word);
				if(wordMap.hasOwnProperty(word))
				{
					activeContext = activeContext.concat(wordMap[word]).unique();
				}
			});
		});
	}
	
	if(activeContext.length == 0)
	{
		if(activeCategory == contextCategory)
		{
			activeCategory = "";
			ShowMenu();
		}
	}
	else if(changeCategory)
	{
		activeCategory = targetCategory;
		ShowMenu();
	}
	else if(activeCategory == contextCategory)
		ShowMenu();
}

// when pressing a button, give the focus back to the prompt field
function KeepFocus(e)
{
	e.preventDefault();
	promptField.focus();
}

function selectCategory(e)
{
	KeepFocus(e);
	// set category from attribute
	activeCategory = e.target.getAttribute("data");
	// rebuild menu
	ShowMenu();
}

function leaveCategory(e)
{
	KeepFocus(e);
	activeCategory = "";
	// rebuild menu
	ShowMenu();
}

// [...]=block "..."=requirement ...=add {|}=cursor {}=insert .,!?;=start
// [{} {|}]
// [,by {}{|}]["by "* and by {}{|}]
// [, {}{|}]

function PatternWalk(text, pattern)
{
	var parts = text.split(/[\,!?;]/);
	var part = parts[parts.length - 1];
	
	var indent = 0;
	var outPattern = "";
	var requirement = ""
	var mode = "";
	var patternFailed = false;
	var partIndex = 0;
	for( let i = 0; i < pattern.length; i++)
	{
		if(mode == "")
		{
			if(pattern[i] == "[")
			{
				indent++;
				mode = "pattern";
				console.log("pattern start:");
			}
		}
		else if(indent > 0)
		{
			if(pattern[i] == "[")
			{
				indent++;
			}
			else if(mode == "pattern")
			{
				if(patternFailed)
				{
					if(pattern[i] == "]")
					{
						indent--;
						if(indent == 0)
						{
							mode = "";
							outPattern = "";
							partIndex = 0;
							patternFailed = false;
							part = parts[parts.length - 1];
						}
					}
					else
					{
					}
				}
				else
				{
					if(pattern[i] == "\"")
					{
						mode = "requirement";
					}
					else if(pattern[i] == "]")
					{
						indent--;
						if(indent == 0)
						{
							mode = "";
							return outPattern;
						}
					}
					else if(pattern[i] == "," || pattern[i] == "!" || pattern[i] == "?" || pattern[i] == ";" )
					{
						let textToCheck = (text+outPattern).trim();
						
						if(textToCheck.endsWith("and"))
						{
							outPattern += "{_}";
							part = "";
							partIndex = 0;
						}
						else if(textToCheck.endsWith("with"))
						{
							outPattern += "{_}";
							part = "";
							partIndex = 0;
						}
						else if(textToCheck.endsWith("of"))
						{
							outPattern += "{_}";
							part = "";
							partIndex = 0;
						}
						else if(textToCheck.endsWith("at"))
						{
							outPattern += "{_}";
							part = "";
							partIndex = 0;
						}
						else if(textToCheck.endsWith("and a"))
						{
							part = "";
							partIndex = 0;
						}
						else if(textToCheck.endsWith("with a"))
						{
							part = "";
							partIndex = 0;
						}
						else if(textToCheck.endsWith("of a"))
						{
							part = "";
							partIndex = 0;
						}
						else if(textToCheck.endsWith("at a"))
						{
							part = "";
							partIndex = 0;
						}
						else if(textToCheck.endsWith("and an"))
						{
							part = "";
							partIndex = 0;
						}
						else if(textToCheck.endsWith("with an"))
						{
							part = "";
							partIndex = 0;
						}
						else if(textToCheck.endsWith("of an"))
						{
							part = "";
							partIndex = 0;
						}
						else if(textToCheck.endsWith("at an"))
						{
							part = "";
							partIndex = 0;
						}
						else if(!textToCheck.endsWith(pattern[i]))
						{
							outPattern += pattern[i];
							part = "";
							partIndex = 0;
						}
					}
					else if(pattern[i] == "{")
					{
						outPattern += pattern[i];
						mode = "write";
					}
					else if(pattern[i] == "." && pattern[i+1] == "*" || pattern[i] == "*")
					{
						let minLength = false;
						if(pattern[i] == "." && pattern[i+1] == "*")
						{
							minLength = true;
							i++;
						}
						var o = pattern.slice(i+1).search(/[^\w\s]/);
						var subpattern = pattern.slice(i+1,i+1+o);
						
						var index = part.lastIndexOf(subpattern);
						var subPatternIndex = subpattern.length;
						while(index == -1)
						{
							if(subPatternIndex <= 1)
							{
								patternFailed = true;
								break;
							}
							
							subPatternIndex--;
							var slice = subpattern.slice(0,subPatternIndex);
							index = part.lastIndexOf(slice);
						}
						if(!patternFailed)
						{
							if(minLength && index == 0)
							{
								patternFailed = true;
							}
							partIndex += index;
						}
						else
						{
						}
					}
					else
					{
						if(partIndex >= part.length)
						{
							outPattern += pattern[i];
						}
						else if(part[partIndex] == pattern[i])
						{
							partIndex++;
						}
						else
						{
							patternFailed = true;
						}
					}
				}
			}
			else if(mode == "requirement")
			{
				if(pattern[i] == "\"")
				{
					if(!part.includes(requirement))
					{
						patternFailed = true;
					}
					else
					{
						partIndex = part.indexOf(requirement)+requirement.length;
					}
					mode = "pattern";
					requirement = "";
				}
				else
				{
					requirement += pattern[i];
				}
			}
			else if(mode == "write")
			{
				if(pattern[i] == "}")
				{
					outPattern += pattern[i];
					mode = "pattern";
				}
				else
				{
					outPattern += pattern[i];
				}
			}
		}
		else if(pattern[i] == "[")
			indent++;
	}
	// fallback
	return ", {}";
}

function InsertPhrase(phrase, pattern)
{
	var text = promptField.value ?? "";
	if(document.activeElement === promptField)
	{
		var pos = getCaretPosition(promptField).end;
		text = promptField.value.slice(0, pos);
	}
	var insert = PatternWalk(text,pattern);
	insert = insert.replace('{}',phrase);
	
	let firstLetter = phrase.trim()[0];
	
	if(firstLetter == "a" || firstLetter == "e" || firstLetter == "i" || firstLetter == "o" || firstLetter == "u")
		insert = insert.replace('{_}',"an");
	else
		insert = insert.replace('{_}',"a");
	
	insert = insert.replace(/{[^|]/,"");
	insert = insert.replace(/[^|]}/,"");
	
	var caret = (text+insert).indexOf("{|}");
	insert = insert.replace('{|}',"");
	// inserting via execCommand is required, this triggers all native browser functionality as if the user wrote into the prompt field.
	parentDoc.execCommand('insertText', false, insert);
	setCaretPosition(promptField, caret, caret);
}

function SelectPhrase(e)
{
	KeepFocus(e);
	var pattern = e.target.getAttribute("pattern");
	var phrase = e.target.getAttribute("data");
	
	InsertPhrase(phrase,pattern);
}

function CheckButtonCondition(condition)
{
	var pos = getCaretPosition(promptField).end;
	var text = promptField.value.slice(0, pos);
	if(condition === "empty")
	{
		return text == "";
	}
}

function ButtonConditions()
{
	conditionalButtons.forEach(entry =>
	{
		let filtered = !CheckButtonCondition(entry.condition);
		
		if(entry.filterGroup != null)
		{
			entry.filterGroup.split(",").forEach( (group) =>
			{
				
				if(activeFilters.includes(group.trim().toLowerCase()))
				{
					filtered = filtered || true;
					return;
				}
			});
		}
		if(filtered)
			entry.element.style.display = "none";
		else
			entry.element.style.display = "inline-block";
	});
}

function ButtonHoverEnter(e)
{
	var text = e.target.getAttribute("tooltip-text");
	var image = e.target.getAttribute("tooltip-image");
	ShowTooltip(text, e.target, image)
}

function ButtonHoverExit(e)
{
	HideTooltip();
}

function ShowTooltip(text, target, image = "")
{
	var cleanedName = image == null? null : image.replace(/[^a-zA-Z0-9 \._\-]/g, '');
	if((text == "" || text == null) && (image == "" || image == null || thumbnails[cleanedName] === undefined))
		return;

	var currentFramePosition = currentFrameAbsolutePosition();
	var rect = target.getBoundingClientRect();
	var element = parentDoc["phraseTooltip"];
	element.innerText = text;
	if(image != "" && image != null && thumbnails[cleanedName] !== undefined)
	{
		
		var img = parentDoc.createElement('img');
		img.src = GetThumbnailURL(cleanedName);
		element.appendChild(img)
	}
	element.style.display = "flex";
	element.style.top = (rect.bottom+currentFramePosition.y)+"px";
	element.style.left = (rect.right+currentFramePosition.x)+"px";
	element.style.width = "inherit";
	element.style.height = "inherit";
}

function base64toBlob(base64Data, contentType) {
    contentType = contentType || '';
    var sliceSize = 1024;
    var byteCharacters = atob(base64Data);
    var bytesLength = byteCharacters.length;
    var slicesCount = Math.ceil(bytesLength / sliceSize);
    var byteArrays = new Array(slicesCount);

    for (var sliceIndex = 0; sliceIndex < slicesCount; ++sliceIndex) {
        var begin = sliceIndex * sliceSize;
        var end = Math.min(begin + sliceSize, bytesLength);

        var bytes = new Array(end - begin);
        for (var offset = begin, i = 0; offset < end; ++i, ++offset) {
            bytes[i] = byteCharacters[offset].charCodeAt(0);
        }
        byteArrays[sliceIndex] = new Uint8Array(bytes);
    }
    return new Blob(byteArrays, { type: contentType });
}

function GetThumbnailURL(image)
{
	if(parentDoc["keyPhraseSuggestionsLoadedBlobs"].hasOwnProperty(image))
	{
		return parentDoc["keyPhraseSuggestionsLoadedBlobs"][image];
	}
	else
	{
		let url = URL.createObjectURL(GetThumbnail(image));
		parentDoc["keyPhraseSuggestionsLoadedBlobs"][image] = url;
		return url;
	}
}

function GetThumbnail(image)
{
	return base64toBlob(thumbnails[image], 'image/webp');
}

function HideTooltip()
{
	var element = parentDoc["phraseTooltip"];
	element.style.display= "none";
	element.innerHTML = "";
	element.style.top = "0px";
	element.style.left = "0px";
	element.style.width = "0px";
	element.style.height = "0px";
}

function RemoveDouble(str, symbol)
{
	let doubleSymbole = symbol+symbol;
	while(str.includes(doubleSymbole))
	{
		str = str.replace(doubleSymbole, symbol);
	}
	return str;
}

function ReplaceAll(str, toReplace, seperator, symbol)
{
	toReplace.split(seperator).forEach( (replaceSymbol) =>
	{
		str = str.replace(replaceSymbol, symbol);
	});
	return str;
}

function WordToKey(word)
{
	if(word.endsWith("s"))
		word = word.slice(0, -1);
	word = word.replace("'", "");
	if(word.endsWith("s"))
		word = word.slice(0, -1);
	word = ReplaceAll(word, "sch;sh;ch;ll;gg;r;l;j;g", ';', 'h');
	word = ReplaceAll(word, "sw;ss;zz;qu;kk;k;z;q;s;x", ';','c');
	word = ReplaceAll(word, "pp;bb;tt;th;ff;p;t;b;f;v", ';','d');
	word = ReplaceAll(word, "yu;yo;oo;u;y;w", ';','o');
	word = ReplaceAll(word, "ee;ie;a;i", ';','e');
	word = ReplaceAll(word, "mm;nn;n", ';','n');
	word = RemoveDouble(word, "l");
	word = RemoveDouble(word, "c");
	word = RemoveDouble(word, "e");
	word = RemoveDouble(word, "m");
	word = RemoveDouble(word, "j");
	word = RemoveDouble(word, "o");
	word = RemoveDouble(word, "d");
	word = RemoveDouble(word, "f");
	return word;
}

Array.prototype.unique = function() {
    var a = this.concat();
    for(var i=0; i<a.length; ++i) {
        for(var j=i+1; j<a.length; ++j) {
            if(a[i] == a[j])
                a.splice(j--, 1);
        }
    }

    return a;
};


function BuildTriggerIndex()
{
	triggerIndex = {};
	wordMap = {};
	tagMap = {};
	for (let category in keyPhrases)
	{
		let count = keyPhrases[category]["entries"].length;
		for(let i = 0; i < count; i++)
		{
			let entry = keyPhrases[category]["entries"][i];
			if(entry["trigger"] != null && entry["trigger"] != "")
			{
				let entryTriggers = entry["trigger"].split(",");
				entryTriggers.forEach( trigger =>
				{
					trigger = trigger.replace(/[^a-zA-Z0-9 \._\-]/g, '').trim().toLowerCase();
					if(!triggers.hasOwnProperty(trigger))
					{
						trigger = WordToKey(trigger);
					}
					if(triggerIndex.hasOwnProperty(trigger))
					{
						triggerIndex[trigger].push( { category: category, index: i });
					}
					else
					{
						triggerIndex[trigger] = [];
						triggerIndex[trigger].push( { category: category, index: i });
					}
				});
			}
			
			/*let words = entry["phrase"].split(" ");
			let wordCount = words.length;
			for(let e = 0; e < wordCount; e++)
			{
				let wordKey = WordToKey(words[e].replace(/[^a-zA-Z0-9 \._\-]/g, '').trim().toLowerCase());
				
				if(wordKey.length < 2)
					continue;
				
				if(!wordMap.hasOwnProperty(wordKey))
				{
					wordMap[wordKey] = [];
				}
				
				let entrySearchTags = entry["search_tags"].split(",");
				entrySearchTags.push(category);
				entrySearchTags.forEach( search_tag =>
				{
					if(search_tag != null && search_tag != "")
					{
						if(search_tag.endsWith("'s"))
							search_tag = search_tag.slice(0, -2);
						if(search_tag.endsWith("s"))
							search_tag = search_tag.slice(0, -1);
						search_tag = search_tag.replace(/[^a-zA-Z0-9 \._\-]/g, '').trim().toLowerCase();
						wordMap[wordKey].push(search_tag);
						if(!tagMap.hasOwnProperty(search_tag))
						{
							tagMap[search_tag] = [];
						}
						tagMap[search_tag].push({ category: category, index: i });
						tagMap[search_tag] = tagMap[search_tag].unique();
					}
				});
				wordMap[wordKey] = wordMap[wordKey].unique();
			}*/
		}
	}
}

function ConditionalButton(entry, button)
{
	if(entry["show_if"] != "" || entry["filter_group"] != "")
		conditionalButtons.push({element:button,condition:entry["show_if"], filterGroup:entry["filter_group"]});
}

// generate menu in suggestion area
function ShowMenu()
{
	// clear all buttons from menu
	ClearSuggestionArea();
	HideTooltip();
	
	// if no chategory is selected
	if(activeCategory == "")
	{
		if(activeContext.length != 0)
		{
			AddButton("Context", selectCategory, "A dynamicly updating category based on the current prompt.", null, null, contextCategory);
		}
		for (var category in keyPhrases)
		{
			AddButton(category, selectCategory, keyPhrases[category]["description"], null, null, category);
		}
		// change iframe size after buttons have been added
		UpdateSize();
	}
	else if(activeCategory == contextCategory)
	{
		// add a button to leave the chategory
		var backbutton = AddButton("&#x2191; back", leaveCategory);
		activeContext.forEach( context =>
		{
			if(tagMap.hasOwnProperty(context))
			{
				var words = tagMap[context].unique();
				words.forEach( word =>
				{
					var entry = keyPhrases[word.category]["entries"][word.index];
					var tempPattern = keyPhrases[word.category]["pattern"];
					
					if(entry["pattern_override"] != "")
					{
						tempPattern = entry["pattern_override"];
					}
					
					var button = AddButton(entry["phrase"], SelectPhrase, entry["description"], entry["phrase"],tempPattern, entry["phrase"]);
					
					ConditionalButton(entry, button);
				});
			}
			if(triggerIndex.hasOwnProperty(context))
			{
				var triggered = triggerIndex[context];
				triggered.forEach( trigger =>
				{
					var entry = keyPhrases[trigger.category]["entries"][trigger.index];
					var tempPattern = keyPhrases[trigger.category]["pattern"];
					
					if(entry["pattern_override"] != "")
					{
						tempPattern = entry["pattern_override"];
					}
					
					var button = AddButton(entry["phrase"], SelectPhrase, entry["description"], entry["phrase"],tempPattern, entry["phrase"]);
					
					ConditionalButton(entry, button);
				});
			}
		});
		
		ButtonConditions();
		// change iframe size after buttons have been added
		UpdateSize();
	}
	// if a chategory is selected
	else
	{
		// add a button to leave the chategory
		var backbutton = AddButton("&#x2191; back", leaveCategory);
		var pattern = keyPhrases[activeCategory]["pattern"];
		keyPhrases[activeCategory]["entries"].forEach(entry =>
		{
			var tempPattern = pattern;
			if(entry["pattern_override"] != "")
			{
				tempPattern = entry["pattern_override"];
			}
			
			var button = AddButton(entry["phrase"], SelectPhrase, entry["description"], entry["phrase"],tempPattern, entry["phrase"]);
			
			ConditionalButton(entry, button);
		});
		ButtonConditions();
		// change iframe size after buttons have been added
		UpdateSize();
	}
}

// listen for clicks on the prompt field
parentDoc.addEventListener("click", (e) =>
{
	// skip if this frame is not visible
	if(!isVisible(frame))
		return;
	
	// if the iframes prompt field is not set, get it and set it
	if(promptField === null)
	{
		GetPromptField();
		ButtonUpdateContext(true);
	}
	
	// get the field with focus
	var target = parentDoc.activeElement;

	// if the field with focus is a prompt field, the %% placeholder %% is set in python
	if(	target.placeholder === placeholder)
	{
		// generate menu
		ShowMenu();
		frame.style.borderBottomWidth = '13px';
	}
	else
	{
		// else hide the iframe
		frame.style.height = "0px";
		frame.style.borderBottomWidth = '0px';
	}
});

function AppendStyle(targetDoc, id, content)
{
	  // get parent document head
	var head = targetDoc.getElementsByTagName('head')[0];

	// add style tag
	var style = targetDoc.createElement('style');
    // set type attribute
	style.setAttribute('type', 'text/css');
	style.id = id;
    // add css forwarded from python
	if (style.styleSheet) {   // IE
        style.styleSheet.cssText = content;
    } else {                // the world
        style.appendChild(parentDoc.createTextNode(content));
    }
	// add style to head
    head.appendChild(style);
}

// Transfer all styles
var head = document.getElementsByTagName("head")[0];
var parentStyle = parentDoc.getElementsByTagName("style");
for (var i = 0; i < parentStyle.length; i++)
	head.appendChild(parentStyle[i].cloneNode(true));
var parentLinks = parentDoc.querySelectorAll('link[rel="stylesheet"]');
for (var i = 0; i < parentLinks.length; i++)
	head.appendChild(parentLinks[i].cloneNode(true));

// add custom style to iframe
frame.classList.add("suggestion-frame");
// clear suggestion area to remove the "javascript failed" message
ClearSuggestionArea();
// collapse the iframe by default
frame.style.height = "0px";
frame.style.borderBottomWidth = '0px';

BuildTriggerIndex();

// only execute once (even though multiple iframes exist)
if(!parentDoc.hasOwnProperty('keyPhraseSuggestionsInitialized'))
{
	AppendStyle(parentDoc, "key-phrase-suggestions", parentCSS);
	
	var tooltip = parentDoc.createElement('div');
	tooltip.id = "phrase-tooltip";
	parentDoc.body.appendChild(tooltip);
	parentDoc["phraseTooltip"] = tooltip;
	// set flag so this only runs once
	parentDoc["keyPhraseSuggestionsLoadedBlobs"] = {};
	parentDoc["keyPhraseSuggestionsInitialized"] = true;
	
	var cssVars = getAllCSSVariableNames();
	computedStyle = getComputedStyle(parentDoc.documentElement);
	
	parentDoc["keyPhraseSuggestionsCSSvariables"] = ":root{";

	cssVars.forEach( (rule) =>
	{
		parentDoc["keyPhraseSuggestionsCSSvariables"] += rule+": "+computedStyle.getPropertyValue(rule)+";";
	});
	parentDoc["keyPhraseSuggestionsCSSvariables"] += "}";
}

AppendStyle(document, "variables", parentDoc["keyPhraseSuggestionsCSSvariables"]);