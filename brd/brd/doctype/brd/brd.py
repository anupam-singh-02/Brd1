# Copyright (c) 2026, anupam and contributors
# For license information, please see license.txt

# import frappe
from frappe.model.document import Document
from frappe.utils import formatdate, money_in_words
from frappe import _
from frappe.utils import nowdate, add_days, cint
from document_compare.validator import validate_document_checklist
import frappe.utils

class BRD(Document):
	pass
