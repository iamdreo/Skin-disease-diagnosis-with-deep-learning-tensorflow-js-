let express = require('express');
let app = express();

app.use(function(req, res, next) {
    console.log('${new Date()} - ${req.method} request for ${req.url}');
    next(); //pass control to the next handler
});

app.use(express.static('./'));

app.listen(process.env.PORT || 8080, () => console.log("All is ok on 8080"));
    