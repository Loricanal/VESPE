function getExtension(filename) {
  var parts = filename.split('.');
  return parts[parts.length - 1];
}


$('form').submit(function() {

	function failValidation(msg) {
	  alert(msg); // just an alert for now but you can spice this up later
	  return false;
	}

	var file1 = $('#inputGroupFile01');
	var ext1 = getExtension(file.val())
	if (ext1 == 'csv') {
	  return failValidation('Please insert a csv file for input data');
	} 

	var file2 = $('#inputGroupFile02');
	var ext2 = getExtension(file.val())
	if (ext2 == 'csv') {
	  return failValidation('Please insert a csv file for target');
	} 


	var file3 = $('#inputGroupFile03');
	var ext3 = getExtension(file.val())
	if (ext3 == 'json') {
	  return failValidation('Please insert a json file for specification');
	} 

	return true;
});