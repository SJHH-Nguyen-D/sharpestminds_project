from flask_wtf import FlaskForm
from wtforms import TextAreaField, StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired, EqualTo, Email, Length, ValidationError, NumberRange


class EmployeePerformanceScoreForm(FlaskForm):
    """ Form class users input their employee demographic information """

    hours_per_week = StringField(
        "How many hours per week do you work?", 
        validators=[
            DataRequired(), 
            NumberRange(
                min=None, 
                max=None, 
                message="Please enter a number within the specified range of 0 - 9999")]
                )
    curr_occ_class = StringField(
        "Current job's occupational 2-digit classification", 
        validators=[
            DataRequired(), 
            NumberRange(
                min=None, 
                max=None, 
                message="Please enter a valid 2-Digit occupational classification")]
                )
    influence = StringField("Indexed measurement of use of influencing capabilities at place of employment (estimated)", 
        validators=[
            DataRequired(), 
            NumberRange(
                min=None, 
                max=None, 
                message="Please enter a valid estimated influence capabilities in at the place of employment.")]
                )
    lifetime_years_of_work = StringField("How many years of paid work have you had in your life time?", 
            validators=[
            DataRequired(),
            NumberRange(
                min=0, 
                max=47, 
                message="Please enter a valid number of years of work that you have worked. If you have worked more than 47 years, please put 47 years.")]
                )
    
    lang_home_deu = BooleanField("What language do you speak at home? Do you speak Dutch?", validators=[DataRequired()])
    lang_home_hun = BooleanField("What language do you speak at home? Do you speak Hungarian?", validators=[DataRequired()])
    cnt_birth_saint_lucia = BooleanField("Was your country of birth Saint Lucia?", validators=[DataRequired()])
    lang_ci_dan = BooleanField("Are you literate in Danish?", validators=[DataRequired()])
    lang_ci_nor = BooleanField("Are you literate in Norwegian?", validators=[DataRequired()])
    curr_status_student = BooleanField("Are you currently a student?", validators=[DataRequired()])
    submit = SubmitField("Sign In")

""" 
    Can be simplified to: what languages do you speak at home (two options), 
    what was your country of birth? What languages are you literate in (two options),
    Are you currently a student/pupil?
"""