$("#file").change(function () {
    let reader= new FileReader();
    reader.onload = function (image) {
    
        $(".col-12").show(0);
        $('#selected-image').attr("src", image.target.result);
        $("#prediction-list").empty();

    }
    
    reader.readAsDataURL(this.files[0]);
    
});

let model;
(async function() {
    model = await tf.loadModel('http://localhost:8080/tfjs-models/model/model.json');
    $('.progress-bar').hide();
})();

$("#predict-button").click(async function () {
    let image = $('#selected-image').get(0);
    let offset = tf.scalar(127.5);
    let tensor = tf.fromPixels(image)
        .resizeNearestNeighbor([224,224])
        .toFloat()
        .sub(offset)
        .div(offset)
        .expandDims();


    let predictions= await model.predict(tensor).data();
    let top5 = Array.from(predictions)
        .map(function (p, i) {
            return {
                probability: p,
                className: IMAGENET_CLASSES[i]
            };
        }).sort(function (a, b) {
            return b.probability - a.probability;
        }).slice(0, 1);

    $("#prediction-list").empty();
    top5.forEach(function (p) {
            $('#prediction-list').append('Disease is most likely: ', p.className + ' with a confidence of ', p.probability.toFixed(2));
        });
    });
    // '<li> ${p.className}: ${p.probability.toFixed(6)} </li>' will be used to display top5 list