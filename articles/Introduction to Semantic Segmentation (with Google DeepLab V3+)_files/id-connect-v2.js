//document.getElementById('signInButton').addEventListener("click", startSigninMainWindow, false);
//document.getElementById('signOutButton').addEventListener("click", startSignoutMainWindow, false);
var redirect_uri = 'https://www.analyticsvidhya.com';
var settings = {
    authority: 'https://id.analyticsvidhya.com/openid',
    client_id: '516577',
    redirect_uri: redirect_uri,
    silent_redirect_uri: redirect_uri+'/back-channel/callback.html',
    post_logout_redirect_uri: redirect_uri,
    response_type: 'id_token token',
    scope: 'openid profile email api',
    accessTokenExpiringNotificationTime: 4,
    automaticSilentRenew: true,
    filterProtocolClaims: true
};
var mgr = new Oidc.UserManager(settings);
var user;
var user2;
var av_is_signed_in = false;


///////////////////////////////
// functions for UI elements
///////////////////////////////

function startSigninMainWindow() {
    mgr.signinRedirect().then(function() {
        //console.log("signinRedirect done");
    }).catch(function(err) {
      //  console.log(err);
    });
}

function startSignoutMainWindow(){
    mgr.signoutRedirect().then(function(resp) {
        // Some state
      //  console.log("signed out", resp);
        //alert('You sure?');
    }).catch(function(err) {
      //  console.log(err);
    });
}
