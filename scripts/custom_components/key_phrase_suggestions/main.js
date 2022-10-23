// parent document
var parentDoc = window.parent.document;
// iframe element in parent document
var frame = window.frameElement;
// the area to put the suggestions in
var suggestionArea = document.getElementById('suggestion_area');
// button height is read when the first button gets created
var buttonHeight = -1;
// the maximum size of the iframe in buttons (3 x buttons height)
var maxHeightInButtons = 3;
// the prompt field connected to this iframe
var promptField = null;
// the category of suggestions
var activeCategory = [];

var conditionalButtons = null;

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
	// apply height to iframe
	frame.style.height = Math.min(suggestionArea.offsetHeight,maxHeight)+"px";
}

// add a button to the suggestion area
function AddButton(label, action, dataTooltip="", tooltipImage="", pattern="", data="")
{
	// create span
	var button = document.createElement("span");
	// label it
	button.innerHTML = label;
	if(data != "")
	{
		// add category attribute to button, will be read on click
		button.setAttribute("data",data);
	}
	if(pattern != "")
	{
		// add category attribute to button, will be read on click
		button.setAttribute("pattern",pattern);
	}
	if(dataTooltip != "")
	{
		// add category attribute to button, will be read on click
		button.setAttribute("tooltip-text",dataTooltip);
	}
	if(tooltipImage != "")
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
			break;
		}
	}
}

function OnChange(e)
{
	ButtonConditions();
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

function SelectPhrase(e)
{
	KeepFocus(e);
	var pattern = e.target.getAttribute("pattern");
	var entry = e.target.getAttribute("data");
	
	// inserting via execCommand is required, this triggers all native browser functionality as if the user wrote into the prompt field.
	parentDoc.execCommand('insertText', false /*no UI*/, pattern.replace('{}',entry));
}

function CheckButtonCondition(condition)
{
	if(condition === "empty")
	{
		return promptField.value == "";
	}
}

function ButtonConditions()
{
	conditionalButtons.forEach(entry =>
	{
		if(CheckButtonCondition(entry.condition))
			entry.element.style.display = "inline-block";
		else
			entry.element.style.display = "none";
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
	if((text == "" || text == null) && (image == "" || image == null || thumbnails[image] === undefined))
		return;

	var currentFramePosition = currentFrameAbsolutePosition();
	var rect = target.getBoundingClientRect();
	var element = parentDoc["phraseTooltip"];
	element.innerText = text;
	if(image != "" && image != null && thumbnails[image] !== undefined)
	{
		
		var img = parentDoc.createElement('img');
		console.log(image);
		img.src = "data:image/webp;base64, "+thumbnails[image];
		
		console.log(thumbnails[image]);
		element.appendChild(img)
	}
	element.style.display = "flex";
	element.style.top = (rect.bottom+currentFramePosition.y)+"px";
	element.style.left = (rect.right+currentFramePosition.x)+"px";
	element.style.width = "inherit";
	element.style.height = "inherit";
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

// generate menu in suggestion area
function ShowMenu()
{
	// clear all buttons from menu
	ClearSuggestionArea();
	HideTooltip();
	
	// if no chategory is selected
	if(activeCategory == "")
	{
		for (var category in keyPhrases)
		{
			AddButton(category, selectCategory, keyPhrases[category]["description"], "", "", category);
		}
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
			
			if(entry["show_if"] != "")
				conditionalButtons.push({element:button,condition:entry["show_if"]});
		});
		// change iframe size after buttons have been added
		UpdateSize();
		ButtonConditions();
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
		GetPromptField();
	
	// get the field with focus
	var target = parentDoc.activeElement;

	// if the field with focus is a prompt field, the %% placeholder %% is set in python
	if(	target.placeholder === placeholder)
	{
		// generate menu
		ShowMenu();
	}
	else
	{
		// else hide the iframe
		frame.style.height = "0px";
	}
});

// add custom style to iframe
frame.classList.add("suggestion-frame");
// clear suggestion area to remove the "javascript failed" message
ClearSuggestionArea();
// collapse the iframe by default
frame.style.height = "0px";

// only execute once (even though multiple iframes exist)
if(!parentDoc.hasOwnProperty('keyPhraseSuggestionsInitialized'))
{
	// get parent document head
	var head = parentDoc.getElementsByTagName('head')[0];
	// add style tag
	var s = parentDoc.createElement('style');
    // set type attribute
	s.setAttribute('type', 'text/css');
    // add css forwarded from python
	if (s.styleSheet) {   // IE
        s.styleSheet.cssText = parentCSS;
    } else {                // the world
        s.appendChild(parentDoc.createTextNode(parentCSS));
    }
	var tooltip = parentDoc.createElement('div');
	tooltip.id = "phrase-tooltip";
	parentDoc.body.appendChild(tooltip);
	parentDoc["phraseTooltip"] = tooltip;
	// add style to head
    head.appendChild(s);
	// set flag so this only runs once
	parentDoc["keyPhraseSuggestionsInitialized"] = true;
}