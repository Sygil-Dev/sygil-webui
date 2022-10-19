// iframe parent
var parentDoc = window.parent.document

// check for mouse pointer locking support, not a requirement but improves the overall experience
var havePointerLock = 'pointerLockElement' in parentDoc ||
  'mozPointerLockElement' in parentDoc ||
  'webkitPointerLockElement' in parentDoc;
  
// the pointer locking exit function
parentDoc.exitPointerLock = parentDoc.exitPointerLock || parentDoc.mozExitPointerLock || parentDoc.webkitExitPointerLock;

// how far should the mouse travel for a step 50 pixel
var pixelPerStep = 50;

// how many steps did the mouse move in as float
var movementDelta = 0.0;
// value when drag started
var lockedValue = 0;
// minimum value from field
var lockedMin = 0;
// maximum value from field
var lockedMax = 0;
// how big should the field steps be
var lockedStep = 0;
// the currently locked in field
var lockedField = null;

parentDoc.addEventListener('mousedown', (e) => {
	// if middle is down
	if(e.button === 1)
	{
		if(e.target.tagName === 'INPUT' && e.target.type === 'number')
		{
			e.preventDefault();
			var field = e.target;
			if(havePointerLock)
				field.requestPointerLock = field.requestPointerLock || field.mozRequestPointerLock || field.webkitRequestPointerLock;

			// save current field
			lockedField = e.target;
			// add class for styling
			lockedField.classList.add("value-dragging");
			// reset movement delta
			movementDelta = 0.0;
			// set to 0 if field is empty
			if(lockedField.value === '')
				lockedField.value = 0;
				
			// save current field value
			lockedValue = parseInt(lockedField.value);
			
			if(lockedField.min === '' || lockedField.min === '-Infinity')
				lockedMin = -99999999;
			else
				lockedMin = parseInt(lockedField.min);
			
			if(lockedField.max === '' || lockedField.max === 'Infinity')
				lockedMax = 99999999;
			else
				lockedMax = parseInt(lockedField.max);
			
			if(lockedField.step === '' || lockedField.step === 'Infinity')
				lockedStep = 1;
			else
				lockedStep = parseInt(lockedField.step);
			
			// lock pointer if available
			if(havePointerLock)
				lockedField.requestPointerLock();
			
			// add drag event
			parentDoc.addEventListener("mousemove", onDrag, false);
		}
	}
});

onDrag = (e) => {
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
};

parentDoc.addEventListener('mouseup', (e) => {
	// if mouse is up
	if(e.button === 1)
	{
		// release pointer lock if available
		if(havePointerLock)
			parentDoc.exitPointerLock();
		
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