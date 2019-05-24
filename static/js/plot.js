function plot(plot_type, id) {

  window.addEventListener('resize', plot_resize);
  // document.getElementById('demo').innerHTML = 'Data: <br>' + plot_type.points;
  Plotly.newPlot(id, plot_type, {}, {responsive:true});

  function plot_resize() {
    var width = document.getElementById(id).clientWidth ;
    // document.getElementById("demo").innerHTML = 'Width: ' + width;
    // var update = {
    //   title: 'Total Width: ' + width,
    //   width: width,
    //   height: 0.9 * window.innerHeight
    //
    // }
    Plotly.relayout(id, {
      width: width,
      height: 0.9 * window.innerHeight
    })
    // Plotly.relayout(id, update)
  };

  // id.on('plotly_hover', plot(plot_type, id) {
  //     var title_change = 'Title Change'
  //     Plotly.relayout(id, {
  //       title: title_change
  //     })
  //   });


}
