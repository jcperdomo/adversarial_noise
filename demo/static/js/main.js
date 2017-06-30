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
                var max = 0;
                var max_idx = 0;
                for (let i = 0; i < 10; i++) {
                    var val = Math.round(data.preds[0][i] * 1000);
                    if (val > max) {
                        max = val;
                        max_idx = i;
                    }
                    var n_digits = String(val).length;
                    for (let j = 0; j < 3 - n_digits; j++) {
                        val = '0' + val;
                    }
                    var text = '0.' + val;
                    if (val > 999) {
                        text = '1.000';
                    }
                    $('#preds tr').eq(i + 1).find('td').eq(0).text(text);
                }
            }
        });
    });

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
        $.ajax({
            // predict class for image
            url: '/api/obfuscate',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(inputs),
            success: function(data) {

                $('#obf_im').attr('src', data.obf_src);

                var max = 0;
                var max_idx = 0;
                for (let i = 0; i < 10; i++) {
                    var val = Math.round(data.preds[0][i] * 1000);
                    if (val > max) {
                        max = val;
                        max_idx = i;
                    }
                    var n_digits = String(val).length;
                    for (let j = 0; j < 3 - n_digits; j++) {
                        val = '0' + val;
                    }
                    var text = '0.' + val;
                    if (val > 999) {
                        text = '1.000';
                    }
                    $('#preds tr').eq(i + 1).find('td').eq(1).text(text);
                }
            }
        });
    });
});
