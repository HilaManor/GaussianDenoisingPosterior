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

    let options = ["faces-514", "faces-34", "faces-213", "faces-227", "parrot", "tiger", "mice"];
    let range_values = ["0.00", "0.12", "0.24", "0.36", "0.48", "0.60", "0.72", "0.84", "0.96", "1.08", "1.20",
                        "1.32", "1.44", "1.56", "1.68", "1.80", "1.92", "2.04", "2.16", "2.28", "2.40", "2.52",
                        "2.64", "2.76", "2.88", "3.00", "-0.12", "-0.24", "-0.36", "-0.48", "-0.60", "-0.72",
                        "-0.84", "-0.96", "-1.08", "-1.20", "-1.32", "-1.44", "-1.56", "-1.68", "-1.80",
                        "-1.92", "-2.04", "-2.16", "-2.28", "-2.40", "-2.52", "-2.64", "-2.76", "-2.88", "-3.00"];

    options.forEach(option => {
        let imageBasePath = "./resources/" + option + "/ev";
        let imagesUrlArray = [];
        for (let ev_i = 1; ev_i <= 5 ; ev_i++) {
            // Less requests:
            if (option.startsWith("faces-514") && ev_i == 5)
                continue;
            else if ((option.startsWith("tiger") || option.startsWith("mice") || option.startsWith("faces")) && ev_i > 3)
                continue;

            let imageBasePath_ev = imageBasePath + String(ev_i) + "_";
            range_values.forEach(rv => {
                imagesUrlArray.push(imageBasePath_ev + rv + ".png");
                if (!option.startsWith("faces"))
                    imagesUrlArray.push(imageBasePath_ev + rv + "_prob.png");
            });
        }
        loadImages(imagesUrlArray).then(images => {
            $("#range-" + option)[0].disabled = false;
            $("#range-" + option)[0].classList.remove("custom-range-disabled");
            $("#range-" + option)[0].classList.add("custom-range");
        });
    });

});

async function loadImages(imageUrlArray) {
    // taken from https://stackoverflow.com/questions/37854355/wait-for-image-loading-to-complete-in-javascript
    const promiseArray = []; // create an array for promises
    const imageArray = []; // array for the images

    for (let imageUrl of imageUrlArray) {

        promiseArray.push(new Promise(resolve => {

            const img = new Image();
            // if you don't need to do anything when the image loads,
            // then you can just write img.onload = resolve;
            img.onload = resolve();  // resolve the promise, indicating that the image has been loaded

            img.src = imageUrl;
            imageArray.push(img);
        }));
    }

    await Promise.all(promiseArray); // wait for all the images to be loaded
    return imageArray;
}



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

function copyBib() {
    let copyText = $("#citation")[0];
    navigator.clipboard.writeText(copyText.getInnerHTML());
}