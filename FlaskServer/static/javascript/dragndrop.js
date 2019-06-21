"use strict";
var dragHandler = function (event) {
    event.preventDefault();
};


var dropHandler = function (event) {
    event.preventDefault();
    var files = event.originalEvent.dataTransfer.files;
    if (files[0].size / 1024 / 1024 > 1) {
        var qrcode = document.getElementById("qrcode");
        qrcode.textContent = "upload file size is limited to 1MB, file too large!";
        return;
    }

    var formData = new FormData();
    formData.append("ics", files[0]);

    var req = {
        url: "/uploadics",
        method: "post",
        processData: false,
        contentType: false,
        data: formData,
        success: (result) => {
            // console.log(files[0]);
            // console.log(result);
            var qrcode = document.getElementById("qrcode");
            if (result.includes("failed")) {
                qrcode.textContent = result;
                return;
            }
            var img = document.createElement('img');
            img.setAttribute("src", result);
            qrcode.textContent = "";
            qrcode.appendChild(img);
        }
    };

    var promise = $.ajax(req);

};

var dropHandlerSet = {
    dragover: dragHandler,
    drop: dropHandler
};

$("body").on(dropHandlerSet);




function notUsed(result) {
    $(".qrcode").html(result);
    //get svg element.
    var svg = document.getElementById("qrcode");

    //get svg source.
    var serializer = new XMLSerializer();
    var source = serializer.serializeToString(svg);

    //add name spaces.
    if (!source.match(/^<svg[^>]+xmlns="http\:\/\/www\.w3\.org\/2000\/svg"/)) {
        source = source.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
    }
    if (!source.match(/^<svg[^>]+"http\:\/\/www\.w3\.org\/1999\/xlink"/)) {
        source = source.replace(/^<svg/, '<svg xmlns:xlink="http://www.w3.org/1999/xlink"');
    }

    //add xml declaration
    source = '<?xml version="1.0" standalone="no"?>\r\n' + source;

    //convert svg source to URI data scheme.
    var url = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(source);

    //set url value to a element's href attribute.
    var link = document.createElement("a");
    var linkText = document.createTextNode("Click for download in HTML5+ or Right click and save this link to save the SVG file");
    link.appendChild(linkText);
    link.href = url;
    link.setAttribute("download", "");
    document.getElementById("right-col").appendChild(link);
    console.log(url);
    //you can download svg file by right click menu.
}
