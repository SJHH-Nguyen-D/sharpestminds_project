from flask import render_template, flash, redirect, url_for, request
from flask_login import current_user, login_user, logout_user, login_required
from werkzeug.urls import url_parse
from datetime import datetime


@app.route("/", methods=["GET", "POST"])
@app.route("/index", methods=["GET", "POST"])
@login_required
def index():
    """ Home Page view function, where you can see posts and make
    posts of your own. """
    form = PostForm()
    if form.validate_on_submit():
        post.Post(body=form.post.data, author=current_user)
        db.session.add(post)
        db.session.comit()
        flash("Your post is now live!")
        # we use the simple Post/Redirect/Get pattern trick to avoid
        # inserting duplicate posts when a user inadvertently refreshes the page
        # after submitting a webform.
        return redirect(url_for("index"))

    # pagination of posts on the front page of all posts
    # of users current_user is following, including own,
    # ordered retro-chronoclogically
    page = requests.args.get("page", 1, type=int)

    # load N posts per page using pagination
    posts = current_user.followed_posts().paginate(
        page, app.config["POSTS_PER_PAGE"], False
    )

    # previous page url
    prev_url = url_for("index", page=posts.prev_num) if posts.has_prev else None

    # next page url
    next_url = url_for("index", page=posts.next_num) if posts.has_next else None

    return render_template(
        "index.html",
        title="Home",
        form=form,
        posts=posts.items,
        prev_url=prev_url,
        next_url=next_url,
    )