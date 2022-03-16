var OPTIONS_FILT = "";

function enableDisableOptions(){
	val_feat = $('#inputGroupFeatures').val()
	val_target = $('#inputGroupTarget').val()
	val_algo = $('#inputGroupAlgo').val()
	val_class = $('#inputGroupClass').val()
	var opt_filt_algo =  OPTIONS_FILT.filter(function(opt) {
		cond1 = (val_feat == "Choose...") || (opt.features_family == val_feat)
		cond2 = (val_target == "Choose...") || (opt.target == val_target)
		cond4 = (val_class == "Choose...") || (opt.class == val_class) 
	    return cond1 && cond2 && cond4;
	});
	let algo_good = opt_filt_algo.map(t=>t.algo);
	$('#inputGroupAlgo option').each(function(){
	    let value = $(this).val()
	    if (algo_good.includes(value)) $("#inputGroupAlgo option[value='" + value + "']").attr("disabled", false);
	    else $("#inputGroupAlgo option[value='" + value + "']").attr("disabled", true);
	});

	var opt_filt_class =  OPTIONS_FILT.filter(function(opt) {
		cond1 = (val_feat == "Choose...") || (opt.features_family == val_feat)
		cond2 = (val_target == "Choose...") || (opt.target == val_target)
		cond3 = (val_algo == "Choose...") || (opt.algo == val_algo)
	    return cond1 && cond2 && cond3;
	});
	let class_good = opt_filt_class.map(t=>t.class);
	$('#inputGroupClass option').each(function(){
	    let value = $(this).val()
	    if (class_good.includes(value)) $("#inputGroupClass option[value='" + value + "']").attr("disabled", false);
	    else $("#inputGroupClass option[value='" + value + "']").attr("disabled", true);
	});
};

function changeValueSelect1() {
	enableDisableOptions()
	if (
		$('#inputGroupFeatures').val() != "Choose..." &&
		$('#inputGroupTarget').val() != "Choose..."
	){
	    $.ajax({
		    url: '/get_data',
		    dataSrc: 'data',
		    type: 'POST',
		    dataType: 'json',
		    contentType: 'application/json',
		    data: JSON.stringify({
		    	"features_family":$('#inputGroupFeatures').val(),
		    	"target":$('#inputGroupTarget').val(),
		    	"algo":$('#inputGroupAlgo').val(),
		    	"class":$('#inputGroupClass').val()
		    }),
		    success: function(data){
		        $('#jds-example').html('');
		        var column_data = '';
		        column_data += '<tr>';
		        for (var key in data[0]){
		            column_data += '<th>' + key + '</th>'
		        };
		        column_data += '</tr>';
		        $('#jds-example').append(column_data),
		        $('th').css({'background-color':'#FFA500', 'color': 'white'});
		        var row_data = '';
		        for (var arr in data){
		            var obj = data[arr];
		            row_data += '<tr>';
		            for (var key in obj){
		                var value = obj[key];
		                row_data += '<td>' + value + '</td>';
		            };
		            row_data += '</tr>'
		        };
		        $('#jds-example').append(row_data);
		        $("#infotable").show();
		        },
		    error: function(data){
		    	$("#infotable").hide();
		        console.log('Error Hit');
		        console.log(data);
		        }
		});
		var enabled = $('#inputGroupClass option:not(:disabled)')
		let cond_class = (enabled.length <= 1) ||  $('#inputGroupAlgo').val()!= "Choose..."
		if (
			$('#inputGroupAlgo').val() != "Choose..." && cond_class
		){
			$('#showStatisticsBtn').prop('disabled', false);
		} else {
			$('#showStatisticsBtn').prop('disabled', true);
		};
	} else {
		$('#showStatisticsBtn').prop('disabled', true);
		$("#infotable").hide();
		$("#infodiv").hide();
	};
};

function changeValueSelect2() {
	if (
		$('#inputGroupSelectFeat1').val() != "Choose..." &&
		$('#inputGroupSelectFeat2').val() != "Choose..."
	){
	    $.ajax({
		    url: '/dependenceplot',
		    type: 'POST',
		    contentType: 'application/json',
		    data: JSON.stringify({
		    	"f1":$('#inputGroupSelectFeat1').find("option:selected").text(),
		    	"f2":$('#inputGroupSelectFeat2').find("option:selected").text()
		    }),
		    success: function(data){
		    	$("#dependenceplot").html(data);
		    },
		    error: function(data){
		        console.log('Error Hit');
		        console.log(data);
		    }
		});
	}
};

function changeValueSelect3() {
	if (
		$('#inputGroupSelectStudents').val() != "Choose..." 
	){
	    $.ajax({
		    url: '/forceplotindividual',
		    type: 'POST',
		    contentType: 'application/json',
		    data: JSON.stringify({
		    	"id_student":$('#inputGroupSelectStudents').val()
		    }),
		    success: function(data){
		    	$("#forceplotindividual").html(data);
			    $.ajax({
				    url: '/decisionplotindividual',
				    type: 'POST',
				    contentType: 'application/json',
				    data: JSON.stringify({
				    	"id_student":$('#inputGroupSelectStudents').val()
				    }),
				    success: function(data){
				    	$("#decisionplotindividual").html(data);
				        },
				    error: function(data){
				        console.log('Error Hit');
				        console.log(data);
				        }
				});
		       },
		    error: function(data){
		        console.log('Error Hit');
		        console.log(data);
		        }
		});
	} 
};

function loadGeneralView(resp){
	$.ajax({
	    url: "/featureimportance1",
	    type: "get",
	   success: function(response) {
	   	$("#infodiv").show();
	   	$("#featureimportance1").html(response);
		$.ajax({
		    url: "/featureimportance2",
		    type: 'get',
		    success: function(response) {
			   	$("#featureimportance2").html(response);
				$.ajax({
				    url: "/featureimportance3",
				    type: "get",
				    success: function(response) {
						$("#featureimportance3").html(response);
						$.ajax({
						    url: "/decisionplot",
						    type: "get",
						   success: function(response) {
						   $("#decisionplot").html(response);
								$.ajax({
								    url: "/forceplot",
								    type: "get",
								   success: function(response) {
								    $("#forceplot").html(response);
								   	loadOptionsFeatures(resp['features']);
									loadOptionsStudents(resp['students']);
								   }
								});
						   }
						});
				    }
				});
		   }
		});	
	   }
	});
};

function loadOptionsFeatures(features){
	let s1 = `<div class="input-group mb-3">
    <div class="input-group-prepend">
      <label class="input-group-text" for="inputGroupSelectFeat1">Feature 1</label>
    </div>
    <select class="custom-select" id="inputGroupSelectFeat1" onchange="changeValueSelect2()">
      <option selected>Choose...</option>`
	features.forEach(function (item, index) {
	  s1 = s1 + "<option value="+item+">"+item+"</option>";
	});
	s1 = s1 + "</select></div>"
	$("#featureinput1").html(s1);
	let s2 = `<div class="input-group mb-3">
    <div class="input-group-prepend">
      <label class="input-group-text" for="inputGroupSelectFeat2">Feature 2</label>
    </div>
    <select class="custom-select" id="inputGroupSelectFeat2" onchange="changeValueSelect2()">
      <option selected>Choose...</option>`
	features.forEach(function (item, index) {
	  s2 = s2 + "<option value="+item+">"+item+"</option>";
	});

	s2 = s2 + "</select></div>"
	$("#featureinput2").html(s2);

};

function loadOptionsStudents(students){
	let s1 = `
	<div class="input-group mb-3">
    <div class="input-group-prepend">
      <label class="input-group-text" for="inputGroupSelectStudents">Student</label>
    </div>
    <select class="custom-select" id="inputGroupSelectStudents" onchange="changeValueSelect3()">
      <option selected>Choose...</option>
    `
	students.forEach(function (item, index) {
	  s1 = s1 + "<option value="+item+">"+item+"</option>";
	});
	s1 = s1 + "</select></div>"
	$("#students").html(s1);
};

$("#showStatisticsBtn").click(function() {
	$.ajax({
		url: "/loadexplainer",
		type: "get",
		data: {},
		success: function(response) {
			loadGeneralView(response);
		}
	});
});


$('#inputGroupFeatures').change(function() {
	changeValueSelect1();
});


$('#inputGroupTarget').change(function() {
	changeValueSelect1();
});


$('#inputGroupAlgo').change(function() {
	changeValueSelect1();
});

$('#inputGroupClass').change(function() {
	changeValueSelect1();
});

$( document ).ready(function() {
	$.ajax({
		url: "/get_options_filter",
		type: "get",
		success: function(response) {
			OPTIONS_FILT = response;
		}
	});
});







