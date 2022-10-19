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
var activeCategory = "";

// check if element is visible
function isVisible(e) {
    return !!( e.offsetWidth || e.offsetHeight || e.getClientRects().length );
}

// remove everything from the suggestion area
function ClearSuggestionArea(text = "")
{
	suggestionArea.innerHTML = text;
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
function AddButton(label, action)
{
	// create span
	var button = document.createElement("span");
	// label it
	button.innerHTML = label;
	// add category attribute to button, will be read on click
	button.setAttribute("category",label);
	// add button function
	button.addEventListener('click', action, false);
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
			break;
		}
	}
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
	activeCategory = e.target.getAttribute("category");
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

// generate menu in suggestion area
function ShowMenu()
{
	// clear all buttons from menu
	ClearSuggestionArea();
	
	// if no chategory is selected
	if(activeCategory === "")
	{
		for (var category in keyPhrases)
		{
			AddButton(category, selectCategory);
		}
		// change iframe size after buttons have been added
		UpdateSize();
	}
	// if a chategory is selected
	else
	{
		// add a button to leave the chategory
		var backbutton = AddButton("&#x2191; back", leaveCategory);
		keyPhrases[activeCategory].forEach(option =>
		{
			AddButton(option, (e) =>
			{
				KeepFocus(e);
				// inserting via execCommand is required, this triggers all native browser functionality as if the user wrote into the prompt field.
				parentDoc.execCommand('insertText', false /*no UI*/, ", "+option);
			});
		});
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
	var s = document.createElement('style');
    // set type attribute
	s.setAttribute('type', 'text/css');
    // add css forwarded from python
	if (s.styleSheet) {   // IE
        s.styleSheet.cssText = parentCSS;
    } else {                // the world
        s.appendChild(document.createTextNode(parentCSS));
    }
	// add style to head
    head.appendChild(s);
	// set flag so this only runs once
	parentDoc["keyPhraseSuggestionsInitialized"] = true;
}