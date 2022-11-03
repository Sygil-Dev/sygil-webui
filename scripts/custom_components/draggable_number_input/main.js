// iframe parent
var parentDoc = window.parent.document

// check for mouse pointer locking support, not a requirement but improves the overall experience
var havePointerLock = 'pointerLockElement' in parentDoc ||
  'mozPointerLockElement' in parentDoc ||
  'webkitPointerLockElement' in parentDoc;
  
// the pointer locking exit function
parentDoc.exitPointerLock = parentDoc.exitPointerLock || parentDoc.mozExitPointerLock || parentDoc.webkitExitPointerLock;

// how far should the mouse travel for a step in pixel
var pixelPerStep = %%pixelPerStep%%;
// how many steps did the mouse move in as float
var movementDelta = 0.0;
// value when drag started
var lockedValue = 0.0;
// minimum value from field
var lockedMin = 0.0;
// maximum value from field
var lockedMax = 0.0;
// how big should the field steps be
var lockedStep = 0.0;
// the currently locked in field
var lockedField = null;

// lock box to just request pointer lock for one element
var lockBox = document.createElement("div");
lockBox.classList.add("lockbox");
parentDoc.body.appendChild(lockBox);
lockBox.requestPointerLock = lockBox.requestPointerLock || lockBox.mozRequestPointerLock || lockBox.webkitRequestPointerLock;

function Lock(field)
{
	var rect = field.getBoundingClientRect();
	lockBox.style.left = (rect.left-2.5)+"px";
	lockBox.style.top = (rect.top-2.5)+"px";
	
	lockBox.style.width = (rect.width+2.5)+"px";
	lockBox.style.height = (rect.height+5)+"px";
	
	lockBox.requestPointerLock();
}

function Unlock()
{
	parentDoc.exitPointerLock();
	lockBox.style.left = "0px";
	lockBox.style.top = "0px";
	
	lockBox.style.width = "0px";
	lockBox.style.height = "0px";
	lockedField.focus();
}

parentDoc.addEventListener('mousedown', (e) => {
	// if middle is down
	if(e.button === 1)
	{
		if(e.target.tagName === 'INPUT' && e.target.type === 'number')
		{
			e.preventDefault();
			var field = e.target;
			if(havePointerLock)
				Lock(field);

			// save current field
			lockedField = e.target;
			// add class for styling
			lockedField.classList.add("value-dragging");
			// reset movement delta
			movementDelta = 0.0;
			// set to 0 if field is empty
			if(lockedField.value === '')
				lockedField.value = 0.0;
				
			// save current field value
			lockedValue = parseFloat(lockedField.value);
			
			if(lockedField.min === '' || lockedField.min === '-Infinity')
				lockedMin = -99999999.0;
			else
				lockedMin = parseFloat(lockedField.min);
			
			if(lockedField.max === '' || lockedField.max === 'Infinity')
				lockedMax = 99999999.0;
			else
				lockedMax = parseFloat(lockedField.max);
			
			if(lockedField.step === '' || lockedField.step === 'Infinity')
				lockedStep = 1.0;
			else
				lockedStep = parseFloat(lockedField.step);
			
			// lock pointer if available
			if(havePointerLock)
				Lock(lockedField);
			
			// add drag event
			parentDoc.addEventListener("mousemove", onDrag, false);
		}
	}
});

function onDrag(e)
{
	if(lockedField !== null)
	{
		// add movement to delta
		movementDelta += e.movementX / pixelPerStep;
		if(lockedField === NaN)
			return;
		// set new value
		let value = lockedValue + Math.floor(Math.abs(movementDelta)) * lockedStep * Math.sign(movementDelta);
		lockedField.focus();
		lockedField.select();
		parentDoc.execCommand('insertText', false /*no UI*/, Math.min(Math.max(value, lockedMin), lockedMax));
	}
}

parentDoc.addEventListener('mouseup', (e) => {
	// if mouse is up
	if(e.button === 1)
	{
		// release pointer lock if available
		if(havePointerLock)
			Unlock();
		
		if(lockedField !== null && lockedField !== NaN)
		{
			// stop drag event
			parentDoc.removeEventListener("mousemove", onDrag, false);
			// remove class for styling
			lockedField.classList.remove("value-dragging");
			// remove reference
			lockedField = null;
		}
	}
});

// only execute once (even though multiple iframes exist)
if(!parentDoc.hasOwnProperty("dragableInitialized"))
{
	var parentCSS =
`
/* Make input-instruction not block mouse events */
.input-instructions,.input-instructions > *{
	pointer-events: none;
	user-select: none;
	-moz-user-select: none;
	-khtml-user-select: none;
	-webkit-user-select: none;
	-o-user-select: none;
}

.lockbox {
	background-color: transparent;
	position: absolute;
	pointer-events: none;
	user-select: none;
	-moz-user-select: none;
	-khtml-user-select: none;
	-webkit-user-select: none;
	-o-user-select: none;
	border-left: dotted 2px rgb(255,75,75);
	border-top: dotted 2px rgb(255,75,75);
	border-bottom: dotted 2px rgb(255,75,75);
	border-right: dotted 1px rgba(255,75,75,0.2);
	border-top-left-radius: 0.25rem;
	border-bottom-left-radius: 0.25rem;
	z-index: 1000;
}
`;
	
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
	parentDoc["dragableInitialized"] = true;
}

