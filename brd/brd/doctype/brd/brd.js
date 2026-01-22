// Copyright (c) 2026, anupam and contributors
// For license information, please see license.txt
//brd.brd.doctype.brd.OnlyOfficeEditor.py
frappe.ui.form.on("BRD", {
	edit_document: async function(frm, cdt, cdn) {
        console.log(">>>>>>>>> custome_edit_document called in document revision")
        if (frm.doc.nda_document) {
            console.log(">>>>>>>>> nda_document exists in document revision -------")
            console.log("Can I read this doc?", frm.perm[0].read);
            console.log("Can I write to this doc?", frm.perm[0].write);
            const file_url = frm.doc.nda_document; // Get file URL from attachment field
            console.log(">>>>>>>>> file_url", file_url)
            await frappe.call({
                method: 'brd.brd.doctype.brd.OnlyOfficeEditor.edit_document',
                args: {
					docname: frm.doc.name,
					file_url: frm.doc.nda_document
				},
                callback: function(r) {
                    if (r.message) {
                        const blob = new Blob([r.message], { type: 'text/html' });
						console.log(blob)
                        const url = URL.createObjectURL(blob);
						console.log(">>>>>>>>> url ", url)
                        window.open(url, '_blank');
                    }
                }
            });
        }
    },
    onload: function(frm) {

        //Setting bpp_id automatically
        if (frm.is_new()) {
            frappe.call({
                method: "brd.brd.doctype.brd.brd.get_next_bpp_id",
                callback: function(r) {
                    frm.set_value("bpp_id", r.message);
                    frm.set_df_property("bpp_id", "read_only", 1);
                }
            });
        } else {
            frm.set_df_property("bpp_id", "read_only", 1);
        }

        //Setting created_by automatically
        if (frm.is_new()) { frappe.call({ method: "brd.brd.doctype.brd.brd.get_Manager_Name", args: { user_email: frappe.session.user }, callback: function(r) { if (r.message) { frm.set_value("created_by", r.message); } } }); }
        
    }
});
