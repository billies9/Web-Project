function plot(plot_type, id) {
  // Will have to get div element to graph correctly
  window.addEventListener('resize', func);
  var width = document.getElementById('container').clientWidth / 2 ;
  var layout = { width: width,
    // width: 0.4 * window.innerWidth,
                height: 0.9 * document.getElementById(id).offsetHeight };

  Plotly.newPlot(id, plot_type, layout, {responsive:true});
  function func() {
    var width = document.getElementById(id).clientWidth ;
    // var width = document.getElementById('container').offsetWidth ;
    Plotly.relayout(id, {
      // width: 0.4 * window.innerWidth,
      width: width,
      height: 0.9 * window.innerHeight
    })
  };

}
