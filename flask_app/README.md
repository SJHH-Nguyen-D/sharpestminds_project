# About

This repo goes through the flask mega tutorial, which can be followed along at: https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world


## Requirements
This project has several package requirements, many of which include flask and the various support extensions including:
* flask-wtf (web forms)
* flask-sqlalchemy (database engine with flask)
* flask-mail (sending emails)
* flask-login (login manager)
* pyjwt (Python JSON Web Tokens for password resetting)
* flask-migrate (database migration engine)
* smtpd (email server transfer protocols)


## Changes to the Database
Making changes to the database, you can migrate the changes with:
```bash
flask db migrate -m "<changed table name>"
flask db upgrade
```

## Using an emulated email server
If you want to set up an emulated email server for debugging, you can set it up with the following command in a terminal:
```bash
python3 -m smtpd -n -c DebuggingServer localhost:8025
```
To configure this emulated server, you need to set two environment variables with:
```bash
export MAIL_SERVER=localhost
export MAIL_PORT=8025
```
If you want to send emails with your Gmail account instead, you can set it up by setting these environment variables instead (if windows, use ```set``` instead of ```export```):
```bash
export MAIL_SERVER=smtp.googlemail.com
export MAIL_PORT=587
export MAIL_USE_TLS=1
export MAIL_USERNAME=<your-gmail-username>
export MAIL_PASSWORD=<your-gmail-password>
```

## Running the application
After having set up the project, run it with:
```bash
flask run
```

## NOTES
* Remove the FLASK_DEBUG=1 environment variable when deploying to production
* In general, to create forms for the website you need:
	* The Form classes in a forms.py file which will be used to dictate what fields the user can enter in the form
	* the route to the form.html in a routes.py file. The routes.py file will redirect the user to the page that they request, as well as load the form that they view in the browser. This also includes all the verification and authentication logic that is required to redirect users to the appropriate page, and plug in all those values into the web page html.
	* A table model in a models.py file if you are using a database. This will store user information that is typically used for the website such as their profile, passwords, posts, etc.
	* the html template form, which is the layout of your page with the placeholder slots, that will be plugged in with your values that will be filled out by the user in the Forms Class in forms.py