document.getElementById('apply-bokeh').addEventListener('click', function() {
    var inputImage = document.getElementById('image-input').files[0];

    if (inputImage) {
        var formData = new FormData();
        formData.append('image', inputImage);

        fetch('/process', {
            method: 'POST',
            body: formData
        })
        .then(response => response.blob())
        .then(data => {
            var outputImage = document.getElementById('output-image');
            outputImage.src = URL.createObjectURL(data);
            outputImage.style.display = 'block';
        });
    }
});
