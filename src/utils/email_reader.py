import imaplib
import email
from email.header import decode_header
import os
from typing import List, Tuple

def connect_imap(server: str, username: str, password: str):
    imap = imaplib.IMAP4_SSL(server)
    imap.login(username, password)
    return imap

def fetch_unseen_emails_with_attachments(server: str, username: str, password: str, folder: str = "INBOX") -> List[Tuple[str, str, List[Tuple[str, bytes]]]]:
    imap = connect_imap(server, username, password)
    imap.select(folder)
    status, messages = imap.search(None, 'UNSEEN')

    emails = []
    for eid in messages[0].split():
        _, msg_data = imap.fetch(eid, "(RFC822)")
        for part in msg_data:
            if isinstance(part, tuple):
                msg = email.message_from_bytes(part[1])
                subject, enc = decode_header(msg["Subject"])[0]
                subject = subject.decode(enc or "utf-8") if isinstance(subject, bytes) else subject

                body = ""
                attachments = []
                if msg.is_multipart():
                    for sub in msg.walk():
                        content_type = sub.get_content_type()
                        disp = str(sub.get("Content-Disposition"))

                        if "attachment" in disp:
                            filename = sub.get_filename()
                            data = sub.get_payload(decode=True)
                            attachments.append((filename, data))
                        elif content_type == "text/plain":
                            body = sub.get_payload(decode=True).decode(errors="ignore")
                else:
                    body = msg.get_payload(decode=True).decode(errors="ignore")

                emails.append((subject, body, attachments))
    imap.logout()
    return emails
