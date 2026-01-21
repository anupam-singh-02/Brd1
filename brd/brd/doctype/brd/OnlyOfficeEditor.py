# document_compare/onlyoffice_editor_api.py
import frappe
import json
import time
import requests
from frappe import _
from frappe.utils import get_url
from frappe.utils.file_manager import save_file

def get_file_url(doc):
    print("Getting file URL for document:", get_url(doc.nda_document))
    return get_url(doc.nda_document)

def get_document_type(file_name):
    extension = file_name.split('.')[-1].lower()
    if extension in ['doc', 'docx', 'odt', 'rtf']:
        return 'word'
    elif extension in ['xls', 'xlsx', 'ods']:
        return 'cell'
    elif extension in ['ppt', 'pptx', 'odp']:
        return 'slide'
    elif extension == 'pdf':
        return 'pdf'
    elif extension == 'vsdx':
        return 'diagram'
    else:
        return 'word'

@frappe.whitelist(allow_guest=False)
def edit_document(file_url, docname):
    try:
        doc = frappe.get_doc("Document Revision", docname)
        version = str(int(time.time()))
        
        unique_key = doc.name
        full_file_url = get_file_url(doc) + f"?v={version}"
        document_type = get_document_type(doc.nda_document)
        callback_url = f"{get_url()}/api/method/document_compare.onlyoffice_editor_api.onlyoffice_callback?document_id={unique_key}"
        
        
        config = {
            "documentType": document_type,
            "document": {
                "fileType": doc.nda_document.split('.')[-1],
                "key": unique_key,
                "title": doc.name,
                "url": full_file_url,
                "trackChanges": True
    
            },
            "editorConfig": {
                "callbackUrl": callback_url,
                "mode": "edit",
                "coEditing": {
                    "mode": "fast",   # REQUIRED for real-time cursors
                    "change": True
                },
                "autosave": False,
                "user": {
                    "id": frappe.session.user,
                    "name": frappe.session.user,
                },
                "permissions": {
                    "edit": True,
                    "review": True
                },
                "customization": {
                    "reviewDisplay": "markup",
                    "forcesave": False,
                    "autosave": False,
                    "Layout": False,
                    "uiTheme": "theme-light",
                    "trackChanges": True,
                    "close": {
                        "visible": True,
                        "text": "Close this file",
                    },
                },
            },
        }
        context = {
            "document": doc,
            "onlyoffice_config": json.dumps(config),
            "onlyoffice_api_js_url": frappe.conf.onlyoffice_api_js_url,
        }
        print("Rendering document edit page with context:", json.dumps(config))
        return frappe.render_template("templates/pages/document_edit.html", context)
    except Exception:
        frappe.log_error(frappe.get_traceback(), "OnlyOffice Editor Error")
        frappe.throw(_("An error occurred while loading the document editor."))

 
import frappe
import os
import json
import requests
from pathlib import Path
from frappe.utils.file_manager import save_file

from frappe.model.document import Document
from frappe.utils import now

@frappe.whitelist(allow_guest=True)
def onlyoffice_callback():
    print("OnlyOffice callback received------------------>")
    try:
        # Parse incoming JSON body
        if frappe.request.content_type == "application/json":
            data = json.loads(frappe.request.get_data(as_text=True))
            print("Received JSON data:", data)
        else:
            data = frappe.form_dict
            print("Received form data:", data)
            
        # -------------------- FIX FOR GUEST USER ---------------------
        # Extract the real user who made changes in OnlyOffice
        actual_user = "Guest"
        if data.get("actions"):
            actual_user = data["actions"][-1].get("userid")
        if not actual_user and data.get("users"):
            actual_user = data["users"][0]
        if not actual_user:
            actual_user = "Guest"
        print("⚡ Running callback as user:", actual_user)
        
        # Override session so logs show correct user
        frappe.set_user(actual_user)
        # --------------------------------------------------------------
        key = data.get("key")
        not_modified = data.get("notmodified")
        # if key:
        #     document_id = "-".join(key.split("-")[:-1])  # Removes the last segment (timestamp)  # Assumes key = "DOCNAME-<timestamp>"
        # else:
        #     document_id = None
        document_id = key
        print("OnlyOffice callback received for document ID:", document_id)
        status = int(data.get("status", 0))
        download_url = data.get("url")
        print("Status:", status, "notmodified:",not_modified ,"Download URL:", download_url)
        
        
        # CASE 1 → No modification (User only viewed or no edits)
        if not_modified in (None, True):
            print(">>> No document change detected. Skipping version creation. <<<")
            return {"error": 0, "message": "No change, version not created"}
                    
        else:
            if status in (2, 3, 6) and download_url:
                response = requests.get(download_url, timeout=60)
                print("Download response status code:", response.status_code)
                if response.status_code == 200:
                    print("Download successful, saving file...")
                    doc = frappe.get_doc("Document Revision", document_id)
                    
                    
                    print("Document found:", doc.name)
                    file_name = f"{document_id}_{now().replace(' ', '_').replace(':', '-')}.docx"
                    saved_file = save_file(file_name, response.content, doc.doctype, doc.name, is_private=0)
                    print("File saved successfully:", saved_file.file_url)
                    print("------------------------------->>:", doc.nda_revisions)
                    
                    doc.revised_nda = doc.nda_document
                    doc.nda_document = saved_file.file_url
                    
                    
                    doc.save(ignore_permissions=True)
                    return {"error": 0, "message": "File saved successfully"}
                else:
                    return {"error": 1, "message": f"Download failed with status {response.status_code}"}
            return {"error": 0, "message": "No action taken (status not in 2, 3, 6 or missing URL)"}
    except Exception as e:
        print("Error in OnlyOffice callback:", str(e))
        return {
            "error": 1,
            "message": str(e),
            "traceback": frappe.get_traceback()
        }