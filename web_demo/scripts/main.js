(function() {
    window.app = {};

    var labeling = null;
    var clearButton = null;
    var drawing = null;

    function initializeUI() {
        document.body.className = '';
        clearButton = document.getElementById('clear-button');
        clearButton.addEventListener('click', clear);
        labeling = document.getElementById('labeling');
        drawing = new window.app.Drawing();
    }

    function clear() {
        drawing.reset();
        document.getElementById("result").style.display = 'none';
        labeling.className = 'hidden';
    }

    window.addEventListener('load', initializeUI);
})();
