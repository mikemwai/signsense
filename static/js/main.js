// Add this to static/js/main.js
document.addEventListener("DOMContentLoaded", function() {
    var menuButton = document.getElementById("menuButton");
    var closeMenuButton = document.getElementById("closeMenuButton");
    var menu = document.getElementById("collapsibleMenu");

    menuButton.addEventListener("click", function() {
        menu.style.left = "0px";
        menuButton.style.display = "none";
    });

    closeMenuButton.addEventListener("click", function(event) {
        event.preventDefault(); // Prevent any default action
        menu.style.left = "-260px";
        menuButton.style.display = "block";
    });
});