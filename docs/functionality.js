// javascript file that set the functionality of some of the components in the index.html file
// imported to the index.html file via the <script> tag in the end of the body part  

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ scroll back to top button ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// javascript code was taken from https://mdbootstrap.com/docs/standard/extended/back-to-top/
//Get the button
let mybutton = document.getElementById("btn-back-to-top");
// When the user scrolls down 20px from the top of the document, show the button
window.onscroll = function () {
scrollFunction();
};
function scrollFunction() {
if (
document.body.scrollTop > 20 ||
document.documentElement.scrollTop > 20
) {
mybutton.style.display = "block";
} else {
mybutton.style.display = "none";
}
}
// When the user clicks on the button, scroll to the top of the document
mybutton.addEventListener("click", backToTop);
function backToTop() {
document.body.scrollTop = 0;
document.documentElement.scrollTop = 0;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ dropdown menu ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$(document).ready(function () {
    $('#dropdown-container- .dropdown-menu').on({
        "click":function(e){
        e.stopPropagation();
    }

});

$('.closer').on('click', function () {
    $('.btn-group').removeClass('open');
    });
});

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ image slider ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// const myRange = document.querySelector('#myRange')
const txtchange = document.querySelector('#txtChange')
const imgchange = document.querySelector('#imgChange')
const texts = [
'First Image', 
'Second Image', 
'Third Image', 
'Fourth Image'
]


$('input[type=range]').on('input', function () {
    // $(this).trigger('change');
    let current_input_id = $(this)[0].id;
    let folder_name = current_input_id.substr(current_input_id.indexOf("-") + 1);
    let newValue = parseFloat($(this)[0].value).toFixed(2);

    let cur_ev = $("#imgChange-" + folder_name)[0].dataset.value;
    swapVizImAndText(folder_name, cur_ev, newValue);
});


function swapVizImAndText(folder_name, cur_ev, newValue) {
    $("#imgChange-" + folder_name)[0].src = "./resources/" + folder_name + "/ev" + cur_ev + "_" + newValue + ".png";
    if ($("#probChange-" + folder_name)[0] != undefined)
        $("#probChange-" + folder_name)[0].src = "./resources/" + folder_name + "/ev" + cur_ev + "_" + newValue + "_prob.png";
    
    if (newValue == "0.00") {
        $("#txtChange-" + folder_name)[0].innerHTML = "\\[\\hat{x} + 0.00 \\sqrt{\\lambda}\\]"; // (MSE-optimal)";
    } else {
        if (newValue.startsWith("-"))
            $("#txtChange-" + folder_name)[0].innerHTML = "\\[\\hat{x} " + newValue + " \\sqrt{\\lambda}\\]";
        else
            $("#txtChange-" + folder_name)[0].innerHTML = "\\[\\hat{x} + " + newValue + " \\sqrt{\\lambda}\\]";
    }
    MathJax.typeset();
}

function change_ev(folder_name, newEV) {
    $('#imgEV-' + folder_name)[0].src = "./resources/" + folder_name + "/pc" + newEV + "_eigvec.png";
    $('#txtEV-' + folder_name)[0].innerHTML = "PC #" + newEV;

    $("#imgChange-" + folder_name)[0].dataset.value = newEV;
    $("#range-" + folder_name)[0].value = 0.00;

    swapVizImAndText(folder_name, newEV, "0.00");
}

window.addEventListener('load', () => {
});