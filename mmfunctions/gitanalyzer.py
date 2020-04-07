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

params_payload = {'is': 'issue', 'state': 'all'}  # alternative states: all, open, closed

logger = logging.getLogger(__name__)
PACKAGE_URL = 'git+https://github.com/sedgewickmm18/mmfunctions.git@'


def get_zen_issues(params, repo_id, zenhub_dict):

    zen_url = 'https://zenhub.ibm.com/p2/workspaces/' + params['ZENHUB_WORKSPACE'] + \
              '/repositories/' + str(repo_id) + '/board'

    kwargs = {
        'headers': {
            'Content-Type': 'application/vnd.github.v3.raw+json',
            'User-Agent': 'Markus zenhub exporter - slightly modified'
        },
        'params': params_payload
    }
    if params['ZENHUB_TOKEN'] != '':
        kwargs['headers']['X-Authentication-Token'] = '%s' % params['ZENHUB_TOKEN']

    print("GET %s" % zen_url)
    resp = requests.get(zen_url, **kwargs)
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


def label_get_component(label):
    component = ''
    if label.startswith('Component') or label.startswith('Scrum') or label.startswith('Squad:'):
        label = label.strip()
        component = label.split(':')[1]
    return component


def label_get_theme(label):
    theme = ''
    if label.find('Theme') >= 0:
        theme = label
    return theme


def label_get_blocked(label):
    blocked = ''
    if label.startswith('blocked'):
        blocked = 'YES'
    return blocked


def label_get_issue_type(label):
    issueType = 'Issue'
    if label.startswith('Epic'):
        issueType = 'Epic'
    elif label.startswith('bug'):
        issueType = 'Bug'
    elif label.startswith('Enhancement'):
        issueType = 'Enhancement'

    return issueType


def label_get_business_value(label):

    # starts with Val or val with optional colon and optional space followed by 1,2,3,4
    if re.fullmatch(label, '^(v|V)al(:|)( [1-4]|[1-4])') is not None:
        return label[-1]
    # default business value is 4
    return 4


def label_get_severity(label):

    # starts with [Ss]ev or [sS]everity with optional colon and optional space followed by 1,2,3,4
    if re.fullmatch(label, '^(s|S)ev(erity|)(:|)( [1-4]|[1-4])') is not None:
        return label[-1]
    # default severity is 3
    return 3


def label_get_risk(label):

    # starts with [Ss]ev or [sS]everity with optional colon and optional space followed by 1,2,3
    if re.fullmatch(label, '^(r|R)isk(:|)( [1-3]|[1-3])', re.I) is not None:
        return label[-1]
    elif re.fullmatch(label, '^risk(:|)( |)easy', re.I) is not None:
        return 3
    elif re.fullmatch(label, '^risk(:|)( |)medium', re.I) is not None:
        return 2
    elif re.fullmatch(label, '^risk(:|)( |)difficult', re.I) is not None:
        return 1
    # default risk is 2
    return 2


def write_issues(params, repo, response, csvout):
    "output a list of issues to csv"
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

        for label in label_list:
            component = label_get_component(label)
            theme = label_get_theme(label)
            blocked = label_get_blocked(label)
            issueType = label_get_issue_type(label)
            businessValue = label_get_business_value(label)
            severity = label_get_severity(label)
            risk = label_get_risk(label)

        csvout.writerow([issue['number'], issue['title'],
                        repo,
                        created_at, updated_at, closed_at,
                        user, assignee, state, milestone,
                        issueType, component,
                        businessValue, severity, risk,
                        theme, blocked, pipeline, str(label_list)])


def get_travis_builds(params, url):
    kwargs = {
        'headers': {
            'Content-Type': 'application/vnd.github.v3.raw+json',
            'User-Agent': 'Padkrish issue exporter - slightly modified'
        },
        'params': params_payload
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
        'params': params_payload
    }
    if params['GITHUB_TOKEN'] != '':
        kwargs['headers']['Authorization'] = 'token %s' % params['GITHUB_TOKEN']

    if url is None:
        url = params['BASE_URL'] + '/api/v3/repos/' + repo + '/issues'

    print("GET %s" % url)
    resp = requests.get(url, **kwargs)
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


def process_all(params):
    # check whether global variables are defined
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
    except Exception as e_ndef:
        logger.error('Global variable not defined: ' + str(e_ndef) + ' ' + str(x))

    csvfilename = 'monitoring-defects.csv'

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
