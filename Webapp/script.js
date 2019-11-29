'use strict';

async function run_entry() {
    try {
        await run();

    } catch (error) {
        log('Error: ' + error);
    }
}

function log(msg) {
    let msg_node = document.getElementById('messages');
    msg_node.appendChild(document.createElement('br'));
    msg_node.appendChild(document.createTextNode(msg));
}

async function loadImage() {
    let imageData = await WebDNN.Image.getImageArray(document.getElementById("image_url").value, {dstW: 224, dstH: 224});
    WebDNN.Image.setImageArrayToCanvas(imageData, 224, 224, document.getElementById('input_image'));

    document.getElementById('run_button').disabled = false;
    log('Image loaded to canvas');
}

function getFrameworkName() {
    return document.querySelector('input[name=framework]:checked').value;
}

var food_model = 'webdnn/output_food_model';
let model_runner = null;
let file_loaded = false;
let model_loaded = false;

function initLoadModel() {
    try {
        loadModel();

    } catch (e) {
        log('ERROR : ' + e);
        console.log(e);
    }
}
    
async function loadModel() {
    let backend_name = document.querySelector('input[name=backend]:checked').value;
    let framework_name = getFrameworkName();
    let backend_key = backend_name + framework_name;
    console.log('Initializing and loading model started.');
    model_runner = await WebDNN.load(food_model, {backendOrder: backend_name});
    console.log(`Loaded backend: ${model_runner.backendName}, version: ${model_runner.descriptor.converted_at}`);

    let loading_msg = document.getElementById('loading_msg');
    loading_msg.innerHTML = 'Choose a photo in your device by clicking the button below, and click <bold>[Test]</bold>!';
    model_loaded = true;

    if(file_loaded){
        document.getElementById('run_button').disabled = false;
    }
}
        
    
async function run() {
    document.getElementById('run_button').disabled = true;
    
    let x = model_runner.inputs[0];
    let y = model_runner.outputs[0];

    let image_options = {
        order: WebDNN.Image.Order.HWC,
        color: WebDNN.Image.Color.BGR,
        bias: [123.68, 116.779, 103.939],
    };

    if (getFrameworkName() === 'chainer' || getFrameworkName() === 'pytorch') {
        image_options.order = WebDNN.Image.Order.CHW;
    }

    if (getFrameworkName() === 'pytorch') {
        image_options.color = WebDNN.Image.Color.RGB;
        image_options.scale = [58.40, 57.12, 57.38];
    }
    
    if (getFrameworkName() === 'keras') {
        image_options.dstW = 224;
        image_options.dstH = 224;
        image_options.order = WebDNN.Image.Order.HWC;
        image_options.color = WebDNN.Image.Color.RGB;
        image_options.bias = [0., 0., 0.];
        image_options.scale = [255., 255., 255.];
    }

    x.set(await WebDNN.Image.getImageArray(document.getElementById('image_url'), image_options));

    let start = performance.now();
    await model_runner.run();
    let elapsed_time = performance.now() - start;


    var thr = 0.99;

    let y_act = y.toActual();

    let predicted_str = "";
    if(y > thr) {
        predicted_str = 'Food';
    } else {
        predicted_str = 'Not-food';
    }
        
    document.getElementById('result').innerHTML = predicted_str;
    
    console.log('prediction : ' + predicted_str);
    console.log('output: ', y_act);
    console.log(`Total Elapsed Time[ms/image]: ${elapsed_time.toFixed(2)}`);

    document.getElementById('run_button').disabled = false;
}

function handleFiles(files) {

    var frame = document.getElementById('frame');
    var filename = document.getElementById('filename');

    if(!files.length){
        file_loaded = false;
    } else {
        file_loaded = true;
    }

    document.getElementById('result').innerHTML = "";

    if(!file_loaded){
        frame.src = 'image-icon.jpg';
        console.log("file_loaded = " + file_loaded);
        document.getElementById('run_button').disabled = true;
        filename.innerHTML = '';
        return;

    } else {
        frame.src = URL.createObjectURL(files[0]);
        console.log("Input file name = " + files[0].name);
        filename.innerHTML = files[0].name;

        if(model_loaded){
            document.getElementById('run_button').disabled = false;
        }
    }

    frame.height = 300;
    frame.onload = function() {
        URL.revokeObjectURL(this.src);
    }
}

window.onload = function () {
 
    var show_image = document.getElementById('image_url');
    var file_button = document.getElementById('file_button');

    console.log('addFileListener()');
    console.log('file_button = ' + file_button);

    // Add listener for file picker
    file_button.addEventListener('click', function(e) {
        if (show_image) {
            show_image.click();
        }
    });

    // Load the model.
    initLoadModel();
}


