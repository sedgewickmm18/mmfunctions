import csv
import re
import requests
import logging
import datetime as dt

"""
Exports Issues from a specified repository to a CSV file
Uses basic authentication (Github username + password) or token to retrieve Issues
from a repository that username has access to. Supports Github API v3.
"""

# params_payload = {'is': 'issue', 'state': 'all'}  # alternative states: all, open, closed

logger = logging.getLogger(__name__)
PACKAGE_URL = 'git+https://github.com/sedgewickmm18/mmfunctions.git@'

# starts with Val or val with optional colon or hyphen and optional space followed by 1,2,3,4
businessvalue_pattern = re.compile('^(v|V)al(:|-|)( [1-4]|[1-4])')
# starts with [Ss]ev or [sS]everity with optional colon or hyphen and optional space followed by 1,2,3,4
severity_pattern = re.compile('^(s|S)ev(erity|)(:|-|)( [1-4]|[1-4])')
# starts with [rR]isk with optional colon or hyphen and optional space followed by 1,2,3
#   or easy, medium or difficult
risk_pattern = re.compile('^(r|R)isk(:|-|)( [1-3]|[1-3])', re.I)
risk_pattern_alt_low = re.compile('^risk(:|-|)( |)low', re.I)
risk_pattern_alt_med = re.compile('^risk(:|-|)( |)medium', re.I)
risk_pattern_alt_high = re.compile('^risk(:|-|)( |)high', re.I)


def get_zen_issues(params, repo_id, zenhub_dict):

    zen_url = 'https://zenhub.ibm.com/p2/workspaces/' + params['ZENHUB_WORKSPACE'] + \
              '/repositories/' + str(repo_id) + '/board'

    kwargs = {
        'headers': {
            'Content-Type': 'application/vnd.github.v3.raw+json',
            'User-Agent': 'Markus zenhub exporter - slightly modified'
        },
        'params': params['GIT_PARAMS']
    }
    if params['ZENHUB_TOKEN'] != '':
        kwargs['headers']['X-Authentication-Token'] = '%s' % params['ZENHUB_TOKEN']

    if params['progress']:
        print("GET %s" % zen_url)

    resp = requests.get(zen_url, **kwargs)

    if params['progress']:
        print("  : => %s" % resp.status_code)

    pipelines = resp.json()['pipelines']

    for p in pipelines:
        p_name = p['name']
        for issue in p['issues']:
            number = issue['issue_number']
            # epic = issue['is_epic']
            # zenhub_dict[number] = (epic, p_name)
            zenhub_dict[number] = p_name

    # import ipdb; ipdb.set_trace()
    if resp.status_code != 200:
        raise Exception(resp.status_code)


def extract_issuefield(person, field):
    if person is not None:
        return person[field]
    return ''


def extract_timevalue(tval, replace=None):
    if tval is not None:
        tval = dt.datetime.strptime(tval, "%Y-%m-%dT%H:%M:%SZ")
    elif replace is not None:
        tval = dt.datetime.strptime(replace, "%Y-%m-%dT%H:%M:%SZ")
    return tval


def labelparm_initialize():
    labelparm = {'component': '', 'theme': '', 'blocked': '',
                 'issueType': 'Issue', 'businessValue': 0,
                 'severity': 3, 'risk': 2}
    return labelparm


def label_get_component(label, labelparm):
    if label.startswith('Component') or label.startswith('Scrum') or label.startswith('Squad:'):
        label = label.strip()
        labelparm['component'] = label.split(':')[1]
    elif labelparm['component'] is None:
        labelparm['component'] = ''


def label_get_theme(label, labelparm):
    if label.find('Theme') >= 0:
        labelparm['theme'] = label
    elif labelparm['theme'] is None:
        labelparm['theme'] = ''


def label_get_blocked(label, labelparm):
    if label.startswith('blocked'):
        labelparm['blocked'] = 'YES'
    elif labelparm['blocked'] is None:
        labelparm['blocked'] = ''


def label_get_issue_type(label, labelparm):
    if label.startswith('Epic'):
        labelparm['issueType'] = 'Epic'
    elif label.startswith('bug'):
        labelparm['issueType'] = 'Bug'
    elif label.startswith('Enhancement'):
        labelparm['issueType'] = 'Enhancement'
    elif labelparm['issueType'] is None:
        labelparm['issueType'] = 'Issue'


def label_get_business_value(label, labelparm):
    if re.fullmatch(businessvalue_pattern, label) is not None:
        labelparm['businessValue'] = label[-1]
    # default business value is 0 (not yet sized)
    elif labelparm['businessValue'] is None:
        labelparm['businessValue'] = 0


def label_get_severity(label, labelparm):
    if re.fullmatch(severity_pattern, label) is not None:
        labelparm['severity'] = label[-1]
    # default severity is 3
    elif labelparm['severity'] is None:
        labelparm['severity'] = 3


def label_get_risk(label, labelparm):
    if re.fullmatch(risk_pattern, label) is not None:
        labelparm['risk'] = label[-1]
    elif re.fullmatch(risk_pattern_alt_low, label) is not None:
        labelparm['risk'] = 3
    elif re.fullmatch(risk_pattern_alt_med, label) is not None:
        labelparm['risk'] = 2
    elif re.fullmatch(risk_pattern_alt_high, label) is not None:
        labelparm['risk'] = 1
    # default risk is 2
    elif labelparm['risk'] is None:
        labelparm['risk'] = 2


def write_issues(params, repo, response, csvout):
    "output a list of issues to csv"
    if params['progress']:
        print("  : Writing %s issues" % len(response.json()))

    for issue in response.json():

        user = extract_issuefield(issue['user'], 'login')
        assignee = extract_issuefield(issue['assignee'], 'login')
        state = issue['state']

        assignees_ = []
        for ass in issue['assignees']:
            assignees_.append(ass['login'])

        milestone = extract_issuefield(issue['milestone'], 'title')
        labels = issue['labels']

        created_at = extract_timevalue(issue['created_at'], '2010-01-01T00:00:00Z')
        updated_at = extract_timevalue(issue['updated_at'])
        closed_at = extract_timevalue(issue['closed_at'])

        label_list = []
        for label in labels:
            label_list.append(str(label['name']).strip().lstrip().rstrip())

        pipeline = ''
        zenhub_dict = params['ZENHUB_DICT']
        try:
            pipeline = zenhub_dict[issue['number']]
        except Exception:
            pass

        labelparm = labelparm_initialize()
        for label in label_list:
            label_get_component(label, labelparm)
            label_get_theme(label, labelparm)
            label_get_blocked(label, labelparm)
            label_get_issue_type(label, labelparm)
            label_get_business_value(label, labelparm)
            label_get_severity(label, labelparm)
            label_get_risk(label, labelparm)

        csvout.writerow([issue['number'], issue['title'],
                        repo,
                        created_at, updated_at, closed_at,
                        user, assignee, state, milestone,
                        labelparm['issueType'], labelparm['component'],
                        labelparm['businessValue'], labelparm['severity'], labelparm['risk'],
                        labelparm['theme'], labelparm['blocked'], pipeline, str(label_list)])


def get_travis_builds(params, url):
    kwargs = {
        'headers': {
            'Content-Type': 'application/vnd.github.v3.raw+json',
            'User-Agent': 'Padkrish issue exporter - slightly modified'
        },
        'params': params['GIT_PARAMS']
    }
    if params['TRAVIS_TOKEN'] != '':
        kwargs['headers']['Authorization'] = 'token %s' % params['TRAVIS_TOKEN']

    #   Travis API 3 doc
    # https://developer.travis-ci.com/resource/builds#Builds
    #
    resp = requests.get('https://api.travis-ci.com/builds')

    # ToDo - no functionality yet
    print(resp)


def get_issues(params, repo=None, url=None):
    kwargs = {
        'headers': {
            'Content-Type': 'application/vnd.github.v3.raw+json',
            'User-Agent': 'Padkrish issue exporter - slightly modified'
        },
        'params': params['GIT_PARAMS']
    }
    if params['GITHUB_TOKEN'] != '':
        kwargs['headers']['Authorization'] = 'token %s' % params['GITHUB_TOKEN']

    if url is None:
        url = params['BASE_URL'] + '/api/v3/repos/' + repo + '/issues'

    if params['progress']:
        print("GET %s" % url)
    resp = requests.get(url, **kwargs)
    if params['progress']:
        print("  : => %s" % resp.status_code)

    if resp.status_code != 200:
        raise Exception(resp.status_code)

    return resp


def next_page(response):
    # more pages? examine the 'link' header returned
    if 'link' in response.headers:
        pages = dict(
            [(rel[6:-1], url[url.index('<')+1:-1]) for url, rel in
                [link.split(';') for link in
                    response.headers['link'].split(',')]])
        if 'last' in pages and 'next' in pages:
            return pages['next']
    return None


def process(params, csvout, repo=None, url=None):
    resp = get_issues(params, repo, url)
    write_issues(params, repo, resp, csvout)
    next_ = next_page(resp)
    if next_ is not None:
        process(params, csvout, repo, next_)


def process_all(params, show_progress=None):

    # default filename
    csvfilename = 'monitoring-defects.csv'

    # check whether global variables are defined and set params to default values
    x = ''
    try:
        x = params['REPO']
        x = params['REPO2']
        x = params['REPO_ID']
        x = params['REPO2_ID']
        x = params['GITHUB_TOKEN']
        x = params['ZENHUB_TOKEN']
        x = params['ZENHUB_WORKSPACE']
        x = params['TRAVIS_TOKEN']
        x = params['BASE_URL']
        if show_progress is None:
            show_progress = False
        params['progress'] = show_progress
        if 'FILENAME' in params and params['FILENAME'] is not None:
            csvfilename = params['FILENAME']
        if 'GIT_PARAMS' not in params or params['GIT_PARAMS'] is None:
            params['GIT_PARAMS'] = {'is': 'issue', 'state': 'all'}  # alternative states: all, open, closed

    except Exception as e_ndef:
        logger.error('Global variable not defined: ' + str(e_ndef) + ' ' + str(x))

    # retrieve zenhub information
    zenhub_dict = {}
    get_zen_issues(params, params['REPO_ID'], zenhub_dict)
    get_zen_issues(params, params['REPO2_ID'], zenhub_dict)
    params['ZENHUB_DICT'] = zenhub_dict

    csvfile = open(csvfilename, 'w', newline='')
    csvout = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
    csvout.writerow(('Title', 'Repo', 'Created', 'Updated', 'Closed', 'Origin', 'Assignee', 'Status', 'Milestone', 'Type',
                     'Component', 'BusinessValue', 'Severity', 'Risk', 'Theme', 'Blocked', 'Pipeline', 'Labels'))
    process(params, csvout, repo=params['REPO'])
    process(params, csvout, repo=params['REPO2'])
    csvfile.close()
