function init_datatables(table_id, user_rand) {
  var currentDate = new Date()
  var day = currentDate.getDate()
  var month = currentDate.getMonth() + 1
  var year = currentDate.getFullYear()
  var hours = currentDate.getHours()
  var minutes = currentDate.getMinutes() + 1
  var morn_night = 'AM'
  var user = ''
  if (hours > 12) {
    hours = hours - 12
    morn_night = 'PM'
  }

  if (user_rand == true) {
    user = 'USER__'
  }
  var d = user + month + '-' + day + '-' + year + '__' + hours + minutes + morn_night;
  $(table_id).DataTable({
    dom:'Bfrtip',
    buttons: ['copy',
              {
                extend:'csv',
                title: d + 'csv'
              },
              {
                extend: 'excel',
                title: d + '.xlsx'
              },
              {
                extend: 'pdf',
                title: d + '.pdf'
              },
              'print'],
    scrollX: true
  });
};
