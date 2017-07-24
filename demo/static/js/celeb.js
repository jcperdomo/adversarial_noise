$(function() {

    // Display uploaded image
    /*
    $('#upload_im').change(function() {
        if (this.files && this.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#orig_im').attr('src', e.target.result);
            }
            reader.readAsDataURL(this.files[0]);
        }
    }); */
    $('#upload_im').change(function() {
        ImageTools.resize(this.files[0], {
            width: 128, // max width
            height: 128 // max height
        }, function(blob, didResize) {
            $('#orig_im').attr('src', window.URL.createObjectURL(blob));
        });
    });

    // Identify person
    $('#orig_im').on('load', function() {
        var im = document.getElementById('orig_im');
        var canvas = document.createElement('canvas');
        canvas.width = im.width;
        canvas.height = im.height;
        canvas.getContext('2d').drawImage(im, 0, 0, im.width, im.height);
        var data = canvas.getContext('2d').getImageData(0, 0, im.width, im.height).data;
        var inputs = [];
        for (var i = 0; i < data.length; i+=4) {
           inputs.push([data[i], data[i+1], data[i+2]]);
        }
        $.ajax({
            // predict class for image
            url: '/illnoise/api/v0.1/identify',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(inputs),
            success: function(data) {
                $('#original_celeb').text(data.celebs[0]);
                $('#original_confidence').text(data.confidences[0]);
            }
        });

        for (var j = 0; j < 10; j++) {
            $('#preds tr').eq(j + 1).find('td').eq(1).text('');
        }
        $('#obf_im').attr('src', '');
        $('#noise_im').attr('src', '');
    });

    // Obfuscation behavior
    $('#obfuscate').on('click', function() {
        var im = document.getElementById('orig_im');
        var canvas = document.createElement('canvas');
        canvas.width = im.width;
        canvas.height = im.height;
        canvas.getContext('2d').drawImage(im, 0, 0, im.width, im.height);
        var data = canvas.getContext('2d').getImageData(0, 0, im.width, im.height).data;
        var inputs = [];
        for (var i = 0; i < data.length; i+=4) {
           inputs.push([data[i], data[i+1], data[i+2]]);
        }

        var dropdown = document.getElementById('target');
        var target = dropdown.selectedIndex - 1; // first option is to let alg decide

        $.ajax({
            // predict class for image
            url: '/illnoise/api/v0.1/celebfuscate',
            method: 'POST',
            contentType: 'application/json',
            //data: JSON.stringify(inputs),
            data: JSON.stringify({ image: inputs, target: target }),
            success: function(data) {

                $('#obf_im').attr('src', data.obf_src);
                $('#noise_im').attr('src', data.noise_src);

                $('#obfuscated_celeb').text(data.celebs[0]);
                $('#obfuscated_confidence').text(data.confidences[0]);
            }
        });
    });
});
