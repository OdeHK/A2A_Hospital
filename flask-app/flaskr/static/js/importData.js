document.getElementById("searchInput").addEventListener("keyup", function() {
          const searchText = this.value.toLowerCase();
          const cards = document.querySelectorAll(".card-doctor");
        
          cards.forEach(card => {
            const text = card.innerText.toLowerCase();
            card.parentElement.style.display = text.includes(searchText) ? "block" : "none";
          });
        });

document.addEventListener("DOMContentLoaded", function() {
    const searchInput = document.getElementById("searchInput");
    const deptFilter = document.getElementById("departmentFilter");
    const hospFilter = document.getElementById("hospitalFilter");
    const cards = document.querySelectorAll(".card-doctor");

    // 🧹 Loại bỏ giá trị trùng trong dropdown
    function removeDuplicateOptions(select) {
    const seen = new Set();
    Array.from(select.options).forEach(opt => {
        if (seen.has(opt.value) && opt.value !== "") opt.remove();
        else seen.add(opt.value);
    });
    }
    removeDuplicateOptions(deptFilter);
    removeDuplicateOptions(hospFilter);

    function filterCards() {
    const searchText = searchInput.value.toLowerCase();
    const deptValue = deptFilter.value.toLowerCase();
    const hospValue = hospFilter.value.toLowerCase();

    cards.forEach(card => {
        const text = card.innerText.toLowerCase();
        const matchSearch = text.includes(searchText);
        const matchDept = deptValue === "" || text.includes(deptValue);
        const matchHosp = hospValue === "" || text.includes(hospValue);

        card.parentElement.style.display = (matchSearch && matchDept && matchHosp) ? "block" : "none";
    });
    }

    searchInput.addEventListener("keyup", filterCards);
    deptFilter.addEventListener("change", filterCards);
    hospFilter.addEventListener("change", filterCards);
});

const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    [...tooltipTriggerList].map(el => new bootstrap.Tooltip(el));

document.addEventListener("DOMContentLoaded", function() {
    const editButtons = document.querySelectorAll(".btn-edit");
    const deleteButtons = document.querySelectorAll(".btn-delete");
    const editForm = document.getElementById("editForm");
    const confirmModal = document.getElementById("confirmDeleteModal");
    const confirmDeleteBtn = document.getElementById("confirmDeleteBtn");
    let deleteRow = null; // lưu id cần xóa
    
    // 📝 EDIT - mở modal chỉnh sửa
    editButtons.forEach(btn => {
        btn.addEventListener("click", () => {
        const card = btn.closest(".card-doctor");
        const name = card.querySelector(".card-title").innerText;
        const age = card.querySelector("p:nth-of-type(1)").innerText.replace("Tuổi: ", "");
        const dept = card.querySelector("p:nth-of-type(2)").innerText.replace("Phòng ban: ", "");
        const hosp = card.querySelector("p:nth-of-type(3)").innerText.replace("Bệnh viện: ", "");
    
        document.getElementById("editRow").value = btn.dataset.id;
        document.getElementById("editName").value = name;
        document.getElementById("editAge").value = age;
        document.getElementById("editDepartment").value = dept;
        document.getElementById("editHospital").value = hosp;
    
        const modal = new bootstrap.Modal(document.getElementById("editModal"));
        modal.show();
        });
    });
    
    // 🗑️ DELETE - mở modal xác nhận custom
    deleteButtons.forEach(btn => {
        btn.addEventListener("click", () => {
        deleteRow = btn.dataset.id;
        const modal = new bootstrap.Modal(confirmModal);
        modal.show();
        });
    });
    
    // ✅ Khi nhấn nút Xóa trong popup
    confirmDeleteBtn.addEventListener("click", () => {
    if (deleteRow) {
        fetch(`/config/delete/${deleteRow}`, { method: "POST" })
        .then(() => {
            const modal = bootstrap.Modal.getInstance(confirmModal);
            modal.hide();

            // 🔔 Hiện toast thông báo
            const toastEl = document.getElementById("deleteToast");
            const toast = new bootstrap.Toast(toastEl, { delay: 3000 });
            toast.show();

            // Reload sau 3.2s để đồng bộ dữ liệu mới
            setTimeout(() => window.location.reload(), 3200);
        });
    }
    });
});

document.addEventListener("DOMContentLoaded", function() {
    const nameElements = document.querySelectorAll(".card-doctor .card-title");

    nameElements.forEach(el => {
    el.style.cursor = "pointer";
    el.addEventListener("click", () => {
        const card = el.closest(".card-doctor");

        // Lấy dữ liệu trong card
        const name = el.innerText;
        const age = card.querySelector("p:nth-of-type(1)").innerText.replace("Tuổi: ", "");
        const department = card.querySelector("p:nth-of-type(2)").innerText.replace("Phòng ban: ", "");
        const hospital = card.querySelector("p:nth-of-type(3)").innerText.replace("Bệnh viện: ", "");
        const note = card.dataset.note || "Chưa có ghi chú.";
        let image = card.dataset.image ? card.dataset.image.trim() : "";
        if (!image || image.toLowerCase() === "none" || image === "null") {
        image = "/static/image/default.jpg";
        }
        document.getElementById("detailImage").src = image;


        // Gán dữ liệu vào modal
        document.getElementById("detailImage").src = image;
        document.getElementById("detailName").innerText = `${name}, ${age} tuổi`;
        document.getElementById("detailDepartment").innerText = `🩺 ${department}`;
        document.getElementById("detailHospital").innerText = `🏥 ${hospital}`;
        document.getElementById("detailNote").innerText = note;

        // Mở modal
        const modal = new bootstrap.Modal(document.getElementById("doctorDetailModal"));
        modal.show();
    });
    });
});