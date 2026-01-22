# Copyright (c) 2026, anupam and contributors
# For license information, please see license.txt

# import frappe
from frappe.model.document import Document
from frappe.utils import formatdate, money_in_words
from frappe import _
from frappe.utils import nowdate, add_days, cint
# from brd.brd.doctype.brd.Validator import validate_document_checklist
import frappe.utils

class BRD(Document):
	pass
	#def before_save(self):
		# print(">>>>>>>>>>> before save", self.nda_document)
		# old_doc = self.get_doc_before_save()
		# print(">>> old doc",old_doc)
		# if not self.is_new() and old_doc and not old_doc.workflow_state == "Draft" and not old_doc.workflow_state == self.workflow_state:
		# 	print(">>>>>>>>>>> state changed")
		# 	roles = frappe.get_roles(frappe.session.user)
		# 	print(">>> roles", roles)
		# 	if not self.revised_nda and 'LEGAL' in roles:
		# 		print(">>>>>>>>>>> revised nda missing")
		# 		return
		# 	if self.revised_nda:
		# 		print(">>>>>>>>>>> Document Changed")
		# 		row = self.append('nda_revisions', {})
		# 		row.original_nda = self.nda_document
		# 		row.nda_archive = self.revised_nda
		# 		row.created_by = self.custom_department_email
		# 		row.change_status = "Document Changed"
		# 		row.department = self.custom_department
		# 		# self.nda_document = self.revised_nda
		# 		self.revised_nda = None
		# 	else:
		# 		print(">>>>>>>>>>> No Change in Document")
		# 		row = self.append('nda_revisions', {})
		# 		row.nda_archive = self.nda_document
		# 		row.created_by = self.custom_department_email
		# 		row.change_status = "No Change"
		# else:
		# 	print(">>> no changes <<<<")


@frappe.whitelist()
def get_next_bpp_id():
	last_id = frappe.db.sql("""
		SELECT MAX(CAST(bpp_id AS UNSIGNED))
		FROM `tabBRD`
	""")[0][0]

	next_id = (last_id or 0) + 1
	return str(next_id)

@frappe.whitelist()
def get_Manager_Name(user_email): 
	role_profile_name = frappe.db.get_value("User", user_email, "role_profile_name") 
	
	if role_profile_name == 'Manager' or 'manager': 
		manager_name = frappe.db.get_value("User", user_email, "name") 
		return manager_name