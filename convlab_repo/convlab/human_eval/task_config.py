# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

task_config = {}

"""A short and descriptive title about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT title appears in search results,
and everywhere the HIT is mentioned.
"""
task_config['hit_title'] = 'Chat and evaluate bot!'

"""A description includes detailed information about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT description appears in the expanded
view of search results, and in the HIT and assignment screens.
"""
task_config['hit_description'] = 'You will chat to a tour information bot and then evaluate that bot.'

"""One or more words or phrases that describe the HIT, separated by commas.
On MTurk website, these words are used in searches to find HITs.
"""
task_config['hit_keywords'] = 'chat,dialog'

"""A detailed task description that will be shown on the HIT task preview page
and on the left side of the chat page. Supports HTML formatting.
"""
task_config['task_description'] = \
    """
    (You can keep accepting new HITs after you finish your current one, so keep working on it if you like the task!)
    <br>
    <b>In this task you will chat with an information desk clerk bot to plan your tour according to a given goal.</b>
    <br>
    For example, your given goal and expected conversation could be: <br><br> 
    <table border="1" cellpadding="10">
    <tr><th>Your goal</th><th>Expected conversation</th></tr>
    <tr><td>
    <ul>
    <li>You are looking for a <b>place to stay</b>. The hotel should be in the <b>cheap</b> price range and should be in the type of <b>hotel</b></li>
    <li>The hotel should include <b>free parking</b> and should include <b>free wifi</b></li>
    <li>Once you find the hotel, you want to book it for <b>6</b> people and <b>3</b> nights</b> starting from <b>tuesday</b></li>
    <li>If the booking fails how about <b>2</b> nights</li>
    <li>Make sure you get the <b>reference number</b></li>
    </ul>
    </td>
    <td>
    <b>You: </b>I am looking for a place to to stay that has cheap price range it should be in a type of hotel<br>
    <b>Info desk: </b>Okay, do you have a specific area you want to stay in?<br>
    <b>You: </b>no, i just need to make sure it's cheap. oh, and i need parking<br>
    <b>Info desk: </b>I found 1 cheap hotel for you that includes parking. Do you like me to book it?<br>
    <b>You: </b>Yes, please. 6 people 3 nights starting on tuesday.<br>
    <b>Info desk: </b>I am sorry but I wasn't able to book that for you for Tuesday. Is there another day you would like to stay or perhaps a shorter stay?<br>
    <b>You: </b>how about only 2 nights.<br>
    <b>Info desk: </b>Booking was successful.\nReference number is : 7GAWK763. Anything else I can do for you?<br>
    <b>You: </b>No, that will be all. Good bye.<br>
    <b>Info desk: </b>Thank you for using our services.<br>
    </td>
    </table>
    <br><br>
    Chat with the bot naturally and stick to your own goal but <b>do not trivially copy the goal descriptions into the message.</b>
    <br>
    Once the conversation is done, you will be asked to rate the bot on metrics like <b>goal accomplishment, language understanding, and response naturalness</b>.
    <br>
    There is a <b>2 min</b> time limit for each turn.
    <br>
    <br>
    - Do not reference the task or MTurk itself during the conversation.
    <br>
    <b><span style="color:red">- No racism, sexism or otherwise offensive comments, or the submission will be rejected and we will report to Amazon.</b></span>
    <br>
    <br>
    """
