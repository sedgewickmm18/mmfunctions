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

squad_pattern= re.compile('^(s|S)quad(:|-|)( |)', re.I)
customer_pattern= re.compile('^(c|C)ustomer(:|-|)( |)', re.I)


FarFuture = '2030-01-01'
IssueSeparator = 10000


#
# get repo_id, repo name from repo nr
#
def get_repo(repo_nr, param):
    if repo_nr == 1:
        return param['REPO_ID'], param['REPO']
    elif repo_nr == 2:
        return param['REPO2_ID'], param['REPO2']
    return 0, ''


def get_repo_nr(repo_id, param):
    if param['REPO_ID'] == repo_id:
        return 1
    elif param['REPO2_ID'] == repo_id:
        return 2
    return 0


def get_full_issue_nr(issue_nr, repo_nr):
    return repo_nr * IssueSeparator + issue_nr


def separate_issue_nr(complex_issue_nr):
    return complex_issue_nr // IssueSeparator, complex_issue_nr % IssueSeparator


#
# retrieve all zenhub releases
#
def get_zen_releases(params, repo_nr, release_dict):

    repo_id, repo = get_repo(repo_nr, params)
    zen_url = params['ZEN_BASE_URL'] + '/p1/repositories/' + str(repo_id) + '/reports/releases'

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

    # print(resp.json())
    for rel in resp.json():
        # print(rel)
        start = dt.datetime.strptime(rel['start_date'][:10], "%Y-%m-%d")
        desired_end = dt.datetime.\
            strptime(rel['desired_end_date'][:10], "%Y-%m-%d")
        closed = FarFuture
        if rel['state'] == 'closed':
            closed = dt.datetime.\
                strptime(rel['closed_at'][:10], "%Y-%m-%d")
        release_dict[rel['release_id']] = (rel['title'], start, desired_end, closed, rel['state'])

    return


#
# build up a hashtable of issue_nr to (release name)
#
def get_zen_release_map(params, release_id, rel_parms, zenhub_rel_dict):

    zen_url = params['ZEN_BASE_URL'] + '/p1/reports/release/' + str(release_id) + '/issues'

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

    # print(resp.json())
    issueList = None
    try:
        issueList = resp.json()
    except Exception:
        print ('Issue with ', resp)
        return

    for iss in resp.json():
        #print (type(iss), ' ' , iss)
        #  print(rel)
        # cheat multi index to single index
        if get_repo_nr(iss['repo_id'], params) == 0:
            continue
        number = get_full_issue_nr(iss['issue_number'], get_repo_nr(iss['repo_id'], params))
        # print (iss['issue_number'], iss['repo_id'], number)
        zenhub_rel_dict[number] = rel_parms

    return


#
# build up a hashtable of issue_nr to (pipeline name, estimate)
#
def get_zen_issues(params, repo_nr, zenhub_dict):

    repo_id, repo = get_repo(repo_nr, params)

    zen_url = params['ZEN_BASE_URL'] + '/p2/workspaces/' + params['ZENHUB_WORKSPACE'] + \
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
            number = get_full_issue_nr(issue['issue_number'], repo_nr)
            estimate = ''
            try:
                estimate = issue['estimate']['value']
            except Exception:
                pass
            # epic = issue['is_epic']
            # zenhub_dict[number] = (epic, p_name)
            zenhub_dict[number] = (p_name, estimate)

    # import ipdb; ipdb.set_trace()
    if resp.status_code != 200:
        raise Exception(resp.status_code)


def extract_issuefield(person, field):
    if person is not None:
        return person[field]
    return ''


def extract_timevalue(params, tval, replace=None):
    timeformat = '%Y-%m-%dT%H:%M:%SZ'
    if tval is not None:
        tval = dt.datetime.strptime(tval, timeformat)
    elif replace is not None:
        tval = dt.datetime.strptime(replace, timeformat)
    if 'JIRA' in params and tval is not None:
        return dt.datetime.strftime(tval, '%d/%b/%y %I:%M %p')
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
        return True
    elif labelparm['component'] is None:
        labelparm['component'] = ''
        return False


def label_get_theme(label, labelparm):
    if label.find('Theme') >= 0 or label == 'AppConnect':
        labelparm['theme'] = label
        return True
    elif labelparm['theme'] is None:
        labelparm['theme'] = ''
        return False


def label_get_blocked(label, labelparm):
    if label.startswith('blocked'):
        labelparm['blocked'] = 'YES'
        return True
    elif labelparm['blocked'] is None:
        labelparm['blocked'] = ''
        return False


def label_get_issue_type(label, labelparm):
    if label.startswith('Epic'):
        labelparm['issueType'] = 'Epic'
        return True
    elif label.startswith('bug'):
        labelparm['issueType'] = 'Bug'
        return True
    elif label.startswith('Enhancement'):
        labelparm['issueType'] = 'Enhancement'
        return True
    elif labelparm['issueType'] is None:
        labelparm['issueType'] = 'Issue'
        return False


def label_get_business_value(label, labelparm):
    if re.fullmatch(businessvalue_pattern, label) is not None:
        labelparm['businessValue'] = label[-1]
        return True
    # default business value is 0 (not yet sized)
    elif labelparm['businessValue'] is None:
        labelparm['businessValue'] = 0
        return False


def label_get_severity(label, labelparm):
    if re.fullmatch(severity_pattern, label) is not None:
        labelparm['severity'] = label[-1]
        return True
    # default severity is 3
    elif labelparm['severity'] is None:
        labelparm['severity'] = 3
        return False


def label_get_risk(label, labelparm):
    if re.fullmatch(risk_pattern, label) is not None:
        labelparm['risk'] = label[-1]
        return True
    elif re.fullmatch(risk_pattern_alt_low, label) is not None:
        labelparm['risk'] = 3
        return True
    elif re.fullmatch(risk_pattern_alt_med, label) is not None:
        labelparm['risk'] = 2
        return True
    elif re.fullmatch(risk_pattern_alt_high, label) is not None:
        labelparm['risk'] = 1
        return True
    # default risk is 2
    elif labelparm['risk'] is None:
        labelparm['risk'] = 2
        return False


def map_user(params, userid):
    if userid in params:
        return params[userid]
    else:
        if 'UNKNOWN_USER' in params:
            return params['UNKNOWN_USER']
    return userid

def write_issues(params, repo_nr, response, csvout):
    "output a list of issues to csv"

    repo_id, repo = get_repo(repo_nr, params)

    if params['progress']:
        print("  : Writing %s issues" % len(response.json()))

    for issue in response.json():

        user = extract_issuefield(issue['user'], 'login')

        number = get_full_issue_nr(issue['number'], repo_nr)

        #url = issue['url']
        url = 'https://github.ibm.com/' + repo + '/issues/' + str(issue['number'])

        assignee = extract_issuefield(issue['assignee'], 'login')
        state = issue['state']

        assignees_ = []
        for ass in issue['assignees']:
            assignees_.append(ass['login'])

        milestone = extract_issuefield(issue['milestone'], 'title')
        labels = issue['labels']

        created_at = extract_timevalue(params, issue['created_at'], '2010-01-01T00:00:00Z')
        updated_at = extract_timevalue(params, issue['updated_at'])
        closed_at = extract_timevalue(params, issue['closed_at'])

        label_list = []
        label_list_copy = []
        for label in labels:
            label_list.append(str(label['name']).strip().lstrip().rstrip())

        pipeline = ''
        estimate = ''
        release = ''
        zenhub_rel_dict = params['ZENHUB_REL_DICT']
        try:
            release = zenhub_rel_dict[number][0]
        except Exception:
            pass

        zenhub_dict = params['ZENHUB_DICT']
        try:
            pipeline = zenhub_dict[number][0]
            estimate = zenhub_dict[number][1]
        except Exception:
            pass

        labelparm = labelparm_initialize()
        for label in label_list:
            t = label_get_component(label, labelparm) or \
                label_get_theme(label, labelparm) or \
                label_get_blocked(label, labelparm) or \
                label_get_issue_type(label, labelparm) or \
                label_get_business_value(label, labelparm) or \
                label_get_severity(label, labelparm) or \
                label_get_risk(label, labelparm)

            # only copy unknown labels to the label list
            if not t:
                label_list_copy.append(label)

        label1, label2, label3, label4, label5 = '','','','',''
        try:
            label1 = label_list_copy.pop(0)
            try:
                label2 = label_list_copy.pop(0)
                try:
                    label3 = label_list_copy.pop(0)
                    try:
                        label4 = label_list_copy.pop(0)
                        try:
                            label5 = label_list_copy.pop(0)
                        except Exception:
                            pass
                    except Exception:
                        pass
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass

        user = map_user(params, user)
        assignee = map_user(params, assignee)

        csvout.writerow([issue['number'], issue['title'],
                        repo, url,
                        created_at, updated_at, closed_at,
                        user, assignee, state, release, milestone,
                        labelparm['issueType'], labelparm['component'], estimate,
                        labelparm['businessValue'], labelparm['severity'], labelparm['risk'],
                        labelparm['theme'], labelparm['blocked'], pipeline,
                        label1, label2, label3, label4, label5, str(label_list_copy)])

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


def get_issues(params, repo_nr=0, url=None):

    repo_id, repo = get_repo(repo_nr, params)

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


def process(params, csvout, repo_nr=1, url=None):
    resp = get_issues(params, repo_nr, url)
    write_issues(params, repo_nr, resp, csvout)
    next_ = next_page(resp)
    if next_ is not None:
        process(params, csvout, repo_nr, next_)


def process_all(params, show_progress=None):

    # default filename
    projectname = ''
    try:
        projectname = params['PROJECT']
    except Exception:
        projectname = 'monitoring'
        pass

    csvfilename = projectname + '-defects.csv'
    releasefilename = projectname + '-releases.csv'
    hasMoreRepos = True

    # check whether global variables are defined and set params to default values
    x = ''
    try:
        x = params['REPO']
        x = params['REPO_ID']
        x = params['GITHUB_TOKEN']
        x = params['ZENHUB_TOKEN']
        x = params['ZENHUB_WORKSPACE']
        x = params['TRAVIS_TOKEN']
        x = params['BASE_URL']
        x = params['ZEN_BASE_URL']
        x = params['IGNORE_RELEASES']
        if show_progress is None:
            show_progress = False
        params['progress'] = show_progress
        if 'FILENAME' in params and params['FILENAME'] is not None:
            csvfilename = params['FILENAME']
        if 'GIT_PARAMS' not in params or params['GIT_PARAMS'] is None:
            params['GIT_PARAMS'] = {'is': 'issue', 'state': 'all'}  # alternative states: all, open, closed

    except Exception as e_ndef:
        logger.error('Global variable not defined: ' + str(e_ndef) + ' ' + str(x))

    try:
        x = params['REPO2']
        x = params['REPO2_ID']
    except Exception:
        hasMoreRepos = False
        pass

    # retrieve zenhub information
    print('loading zenhub releases')
    zenhub_releases = {}
    get_zen_releases(params, 1, zenhub_releases)

    if hasMoreRepos:
        get_zen_releases(params, 2, zenhub_releases)

    params['ZENHUB_RELEASES'] = zenhub_releases

    print('loading zenhub issues per releases')
    zenhub_rel_dict = {}
    csvfile = open(releasefilename, 'w', newline='')
    csvout = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
    csvout.writerow(('Id', 'Title', 'Start', 'DesiredEnd', 'Closed'))
    for rel in params['ZENHUB_RELEASES']:
        # ignore releases by name or state
        if zenhub_releases[rel][0] in params['IGNORE_RELEASES'] or zenhub_releases[rel][4] in params['IGNORE_RELEASES']:
            continue
        csvout.writerow([rel, zenhub_releases[rel][0], zenhub_releases[rel][1], zenhub_releases[rel][2], zenhub_releases[rel][3]])
        get_zen_release_map(params, rel, params['ZENHUB_RELEASES'][rel], zenhub_rel_dict)
    csvfile.close()
    params['ZENHUB_REL_DICT'] = zenhub_rel_dict

    # print(zenhub_rel_dict)
    # return

    print('loading zenhub issues by board')
    zenhub_dict = {}
    get_zen_issues(params, 1, zenhub_dict)

    if hasMoreRepos:
        get_zen_issues(params, 2, zenhub_dict)

    params['ZENHUB_DICT'] = zenhub_dict

    csvfile = open(csvfilename, 'w', newline='')
    csvout = csv.writer(csvfile, delimiter='|', quoting=csv.QUOTE_MINIMAL)
    csvout.writerow(('IssueNr', 'Title', 'Repo', 'Url', 'Created', 'Updated', 'Closed', 'Origin', 'Assignee', 'Status', 'Release',
                     'Milestone', 'Type', 'Component', 'Estimate', 'BusinessValue', 'Severity', 'Risk', 'Theme', 'Blocked', 'Pipeline',
                     'Label1', 'Label2', 'Label3', 'Label4', 'Label5', 'Labels'))
    print('Process github repo 1')
    process(params, csvout, repo_nr=1)

    if hasMoreRepos:
        print('Process github repo 2')
        process(params, csvout, repo_nr=2)

    csvfile.close()
