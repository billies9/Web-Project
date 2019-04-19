window.onload = function plot() {
  // Will have to get div element to graph correctly
  window.addEventListener('resize', func);
  var graphs = {{ price | safe }};
  var width = document.getElementById('price').offsetWidth;
  var layout = { width: width,
    // width: 0.4 * window.innerWidth,
                height: 0.9 * document.getElementById('price').offetHeight };

  Plotly.newPlot('price', graphs, layout, {responsive:true});
  function func() {
    var width = document.getElementById('price').offsetWidth;
    Plotly.relayout('price', {
      // width: 0.4 * window.innerWidth,
      width: width,
      height: 0.9 * window.innerHeight
    })
  };

  window.addEventListener("resize", myFunction);

  var x = 0;
  function myFunction() {
    // var txt = x += 1;
    var width = document.getElementById('price').offsetWidth;
    document.getElementById("demo1").innerHTML = "Width " + width;
  };

}
