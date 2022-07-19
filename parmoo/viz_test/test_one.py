# import time
# from dash.testing.application_runners import import_app
# from selenium import webdriver
# driver = webdriver.Chrome()


# def test_ppaaa001(dash_duo):

#     app = import_app("viz_test.dummy")
#     dash_duo.start_server(app)

#     dash_duo.wait_for_text_to_equal("#subheader",
#                                     "Interact with your MOOP results",
#                                     timeout=100,)

#     assert dash_duo.get_logs() == [], "Browser console should contain no error"

#     return None

# 1. imports of your dash app
import dash
from dash import html
# 2. give each testcase a tcid, and pass the fixture
# as a function argument, less boilerplate
def test_bsly001_falsy_child(dash_duo):
    # 3. define your app inside the test function
    app = dash.Dash(__name__)
    app.layout = html.Div(id="nully-wrapper", children=0)
    # 4. host the app locally in a thread, all dash server configs could be
    # passed after the first app argument
    dash_duo.start_server(app)
    # 5. use wait_for_* if your target element is the result of a callback,
    # keep in mind even the initial rendering can trigger callbacks
    dash_duo.wait_for_text_to_equal("#nully-wrapper", "0", timeout=4)
    # 6. use this form if its present is expected at the action point
    assert dash_duo.find_element("#nully-wrapper").text == "0"
    # 7. to make the checkpoint more readable, you can describe the
    # acceptance criterion as an assert message after the comma.
    assert dash_duo.get_logs() == [], "browser console should contain no error"
    # 8. visual testing with percy snapshot
    dash_duo.percy_snapshot("bsly001-layout")