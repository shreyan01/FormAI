const express=require("express");
const path=require("path");
const app=express();
const port=80;
const fs=require("fs");
const { title } = require("process");

app.use("/static", express.static('static'))
app.use(express.urlencoded())

app.get("/", (req, res)=>{
    res.status(200).render('index.html',title);
});

app.listen(port, ()=>{
    `The app started successfully on ${port}`
})