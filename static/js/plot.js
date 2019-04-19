function plot(plot_type, id) {

  window.addEventListener('resize', func);
  
  Plotly.newPlot(id, plot_type, {}, {responsive:true});
  function func() {
    var width = document.getElementById(id).clientWidth ;
    Plotly.relayout(id, {
      width: width,
      height: 0.9 * window.innerHeight
    })
  };

}
