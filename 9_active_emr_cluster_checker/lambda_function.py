import http.client
import json
import logging
import urllib.parse

import boto3

_logger = logging.getLogger()
_logger.setLevel(logging.INFO)

SLACK_WEBHOOK_URLPATH = "https://hooks.slack.com/services/XXXXXXXXX/XXXXXXXXX/XXXXXXXXXXXXXXXXXXXXXXXXX"
SLACK_NOTIFICATION_CHANNEL = "@thom"  # prefix @ for users, # for groups


class ActiveEMRClusterChecker:
    logger = _logger

    def __init__(self):
        self.emr_client = None
        self.active_cluster_ids = None

    def run(self):
        self._set_emr_client()
        self._list_active_clusters()
        self._log_number_of_active_clusters()
        self._send_slack_notification_for_each_active_cluster()
        self._terminate_active_clusters()

    def _set_emr_client(self):
        session = boto3.Session()
        self.emr_client = session.client('emr')

    def _list_active_clusters(self):
        active_cluster_states = ['STARTING', 'BOOTSTRAPPING', 'RUNNING', 'WAITING']
        response = self.emr_client.list_clusters(ClusterStates=active_cluster_states)
        self.active_cluster_ids = [cluster["Id"] for cluster in response["Clusters"]]

    def _log_number_of_active_clusters(self):
        if not self.active_cluster_ids:
            self.logger.info("No active clusters...")
        else:
            self.logger.info(f"Found {len(self.active_cluster_ids)} active clusters...")

    def _send_slack_notification_for_each_active_cluster(self):
        for cluster_id in self.active_cluster_ids:
            self._send_slack_notification_for_active_cluster(cluster_id)

    def _send_slack_notification_for_active_cluster(self, cluster_id):
        description = self._describe_cluster(cluster_id)
        message = self._get_slack_message_from_description(description)
        icon = self._get_icon_emoji_based_on_description(description)
        username = self._get_username(description)
        self._send_slack_notification(message, icon, username)

    def _describe_cluster(self, cluster_id):
        description = self.emr_client.describe_cluster(ClusterId=cluster_id)
        state = description['Cluster']['Status']['State']
        name = description['Cluster']['Name']
        keypair = description['Cluster']['Ec2InstanceAttributes']['Ec2KeyName']
        description = {'state': state, 'name': name, 'keypair': keypair}
        return description

    def _get_slack_message_from_description(self, description):
        message = "Cluster `{name}` was still active in state `{state}` with keypair `{keypair}`. " \
                  .format(state=description['state'], name=description['name'], keypair=description['keypair'])
        self.logger.info(f"Message: {message}")
        return message

    def _get_icon_emoji_based_on_description(self, description):
        keypair = self._get_keypair(description)
        if keypair == "thom":
            return ":thom:"
        else:
            return ":money_with_wings:"

    def _get_username(self, description):
        keypair = self._get_keypair(description)
        username = f"Active EMR Cluster Bot ({keypair})"
        return username

    @staticmethod
    def _get_keypair(description):
        return description["keypair"]

    @staticmethod
    def _send_slack_notification(message, icon, username):
        slack_notifier = SlackNotifier()
        slack_notifier.send_message(message, icon, username)

    def _terminate_active_clusters(self):
        self.emr_client.terminate_job_flows(
            JobFlowIds=self.active_cluster_ids
        )
        self.logger.info("Terminated all active clusters...")


class SlackNotifier:
    logger = _logger

    def __init__(self):
        self.channel = SLACK_NOTIFICATION_CHANNEL
        self.slack_webhook_urlpath = SLACK_WEBHOOK_URLPATH

    def send_message(self, message, icon, username):
        payload = self._get_payload(username, icon, message)
        data = self. _get_encoded_data_object(payload)
        headers = self._get_headers()
        response = self._send_post_request(data, headers)
        self._log_response_status(response)

    def _get_payload(self, username, icon, message):
        payload_dict = {
            'channel': self.channel,
            'username': username,
            'icon_emoji': icon,
            'text': message,
            "attachments": [
                {
                    "color": "#36a64f",
                    "title": "Shame. Shame. Shame.",
                    "image_url": "https://media.giphy.com/media/Ob7p7lDT99cd2/giphy.gif",
                    "thumb_url": "https://media.giphy.com/media/m6tmCnGCNvTby/giphy.gif",
                }
            ]
        }
        payload = json.dumps(payload_dict)
        return payload

    @staticmethod
    def _get_encoded_data_object(payload):
        values = {'payload': payload}
        str_values = {}
        for k, v in values.items():
            str_values[k] = v.encode('utf-8')
        data = urllib.parse.urlencode(str_values)
        return data

    @staticmethod
    def _get_headers():
        headers = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain"}
        return headers

    def _send_post_request(self, body, headers):
        https_connection = self._get_https_connection_with_slack()
        https_connection.request('POST', self.slack_webhook_urlpath, body, headers)
        response = https_connection.getresponse()
        return response

    @staticmethod
    def _get_https_connection_with_slack():
        h = http.client.HTTPSConnection('hooks.slack.com')
        return h

    def _log_response_status(self, response):
        if response.status == 200:
            self.logger.info("Succesfully send message to Slack.")
        else:
            self.logger.critical("Send message to Slack failed with "
                                 "status code '{}' and reason '{}'.".format(response.status, response.reason))


def lambda_handler(event, context):
    ActiveEMRClusterChecker().run()
