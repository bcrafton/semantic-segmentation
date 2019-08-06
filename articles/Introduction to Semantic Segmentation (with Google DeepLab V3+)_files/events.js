function getAptStaticURL() {
    let prodHosts = ["www.analyticsvidhya.com", "trainings.analyticsvidhya.com",
                     "datahack.analyticsvidhya.com", "discuss.analyticsvidhya.com", "id.analyticsvidhya.com",
                     "brahma.analyticsvidhya.com", "courses.analyticsvidhya.com"];
    let staticProdURL = "https://brahma.analyticsvidhya.com/static/CACHE";
    let staticLocalURL = "http://localhost:8000/static/CACHE";
    let staticDevURL = "https://brahma.aifest.org/static/CACHE";

    let currentHost = window.location.hostname;

    if (prodHosts.indexOf(currentHost) != -1) {
        return staticProdURL;
    } else if (currentHost == "localhost" || currentHost == "localhost:8000") {
        return staticLocalURL;
    } else {
        return staticDevURL;
    }
}

// console.log(window.location.hostname);

let manifestFilename = "manifest.json";

let staticURL = getAptStaticURL();

let staticManifestURL = staticURL + '/' + manifestFilename;

function getScript(url, callback) {
    var script = document.createElement('script');
    script.type = 'text/javascript';
    script.src = url;
    // most browsers
    script.onload = callback;
    // IE 6 & 7
    script.onreadystatechange = function() {
        if (this.readyState == 'complete') {
            callback();
        }
    };
    document.getElementsByTagName('head')[0].appendChild(script);
}


$.ajax({
    dataType: "json",
    url: staticManifestURL,
    success: function (data) {
        let URLString = Object.values(data)[0];
        let analyticsFilename = URLString.match(/CACHE\/js\/(.*?)\.js\">/i)[1] + '.js';

        // console.log(staticManifestURL, analyticsFilename);

        let analyticsJSURL = staticURL + '/js/' + analyticsFilename;

        getScript(analyticsJSURL, function () {
            console.log("Events loaded");
        });
    }
});
