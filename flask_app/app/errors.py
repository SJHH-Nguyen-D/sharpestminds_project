from flask import render_template

@app.errorhandler(404)
def not_found_error(error):
    """ Redirects user to page-not-found page for code 404 """
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_error(error):
    """ Redirects user to internal error page for code 500 """
    return render_template("500.html"), 500
