function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function (e) {
            $('#orig_im').attr('src', e.target.result);
        }
        reader.readAsDataURL(input.files[0]);
    }
}

$(function() {
    $('#upload_btn').on('click', function() {
        var fd = new FormData();
        fd.append('file', $('input[type=file]')[0].files[0]);
        $.ajax({
            url: '/api/upload',
            method: 'POST',
            data: fd,
            cache: false,
            contentType: false,
            processData: false
        });
    });

    $('#upload_im').change(function() {
        //readURL(this);
        if (this.files && this.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#orig_im').attr('src', e.target.result);
            }
            reader.readAsDataURL(this.files[0]);
        }
    });

    $('#orig_im').on('load', function () {
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
            url: '/api/predict',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(inputs),
            success: function(data) {

            }
        });
    });

    $('#obfuscate').on('click', function() {
        $.ajax({
            url: '/api/obfuscate',
            method: 'POST',
            contentType: 'TODO',
            data: 'TODO',
            success: (data) => {
                // put the results in the table
            }
        });
    });
});
